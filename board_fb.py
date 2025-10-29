#!/usr/bin/env python3
import os, time, mmap, fcntl, struct, requests, sys
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

# ---------------- Config ----------------
# City bus stop in front of the house
BUS_STATION = "Winterthur, Else Züblin"
# Train station behind the house
TRAIN_STATION = "Winterthur Hegi (SBB)"
# We only care about trains heading toward the main station
TARGET_DESTINATION = "Winterthur, Hauptbahnhof"
# Display dimensions - Portrait orientation (320x480)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 320, 480  # Portrait orientation
LIMIT = 12
REFRESH_SECS = 30

# Framebuffer device to use
FRAMEBUFFER_DEVICE = "/dev/fb1"  # Change to /dev/fb0, /dev/fb1, etc. as needed

# Font sizes for portrait display
FONT_REG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
FONT_MED = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FONT_BIG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

# No rotation needed - we'll work directly in portrait mode
ROTATE_IF_LANDSCAPE = 0

# Try setting this to True if colors look wrong or you see "broken icons"
BGR565 = True  # Changed to True to fix violet/green color distortion

# Optional: draw a one-time test pattern on first frame to verify visibility
DRAW_TEST_PATTERN_ONCE = False

# Rotation direction when framebuffer is landscape: -90 (clockwise) or +90 (counter-clockwise)
LANDSCAPE_ROTATE_DEG = -90

# -------------- fb ioctls ---------------
FBIOGET_VSCREENINFO = 0x4600

def get_fbinfo(fd):
    buf = bytearray(160)
    fcntl.ioctl(fd, FBIOGET_VSCREENINFO, buf, True)
    xres, yres = struct.unpack_from("<II", buf, 0)
    bpp = struct.unpack_from("<I", buf, 32)[0]
    return {"xres": xres, "yres": yres, "bpp": bpp}

def get_stride_default(xres):
    # Fallback: 2 Bytes pro Pixel
    return xres * 2

def get_stride_sysfs():
    try:
        # Extract fb number from device path (e.g., "/dev/fb1" -> "fb1")
        fb_name = os.path.basename(FRAMEBUFFER_DEVICE)
        with open(f"/sys/class/graphics/{fb_name}/stride", "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

# -------------- helpers -----------------

def clear_framebuffer(mm, xres, yres, stride):
    """Clear the entire framebuffer with black pixels"""
    print("Clearing framebuffer...")
    black_pixel = b'\x00\x00'  # Black in RGB565/BGR565
    row_data = black_pixel * xres
    
    for y in range(yres):
        mm.seek(y * stride)
        mm.write(row_data)
    mm.flush()
    print("Framebuffer cleared")

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    # Pillow >=10: textbbox statt textsize
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t

def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l

def rgb888_to_rgb565(img: Image.Image) -> bytes:
    import numpy as np
    arr = np.asarray(img, dtype=np.uint8)
    r = (arr[:, :, 0] >> 3).astype('uint16')
    g = (arr[:, :, 1] >> 2).astype('uint16')
    b = (arr[:, :, 2] >> 3).astype('uint16')
    if BGR565:
        rgb565 = (b << 11) | (g << 5) | r
    else:
        rgb565 = (r << 11) | (g << 5) | b
    return rgb565.byteswap().tobytes()

def line_color(cat, name):
    c = (cat or "").upper(); n = (name or "").upper()

    # S-Bahn
    if c.startswith("S") or n.startswith("S"): return (65,120,255)

    # Tram
    if c in ("TRAM", "T"):                  return (0,170,170)

    # Bus/Nachtbus
    if c in ("BUS","N","SN") or (n and n[0].isdigit()):
        return (60,160,220)

    # Fernverkehr
    if c in ("IC","IR","RE","RJ","RJX","EN"): return (60,200,120)
    return (130,130,130)

def _format_departure_item(it, now_utc):
    stop = it.get("stop", {})
    dep_iso = stop.get("departure")
    if not dep_iso:
        return None
    dep = datetime.fromisoformat(dep_iso.replace("Z", "+00:00"))
    delta = dep - now_utc
    mins = max(0, int(delta.total_seconds() // 60))
    secs = max(0, int(delta.total_seconds() % 60))
    delay = int(stop.get("delay", 0) or 0)
    when_local = dep.astimezone()
    cat = it.get("category") or ""
    name = it.get("name") or (cat + str(it.get("number", "")))
    to = it.get("to", "")
    return {
        "time": when_local.strftime("%H:%M"),
        "mins": mins,
        "secs": secs,
        "delay": delay,
        "line": name,
        "cat": cat,
        "to": to,
        "_epoch": dep.timestamp(),
    }

def fetch_bus_departures(station, limit):
    r = requests.get(
        "https://transport.opendata.ch/v1/stationboard",
        params={"station": station, "limit": limit} # , "transportations[]": ["bus"]},
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    now_utc = datetime.now(timezone.utc)
    out = []
    for it in data.get("stationboard", []):
        item = _format_departure_item(it, now_utc)
        if not item:
            continue
        # Keep only bus-like items (defensive; API should already filter)
        cat_u = (item["cat"] or "").upper()
        name_u = (item["line"] or "").upper()
        if cat_u in ("BUS", "N", "SN") or (name_u and name_u[0:1].isdigit()):
            out.append(item)
    return out

def fetch_train_departures_towards(station, limit, target_destination):
    r = requests.get(
        "https://transport.opendata.ch/v1/stationboard",
        params={"station": station, "limit": limit},
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    now_utc = datetime.now(timezone.utc)
    out = []
    for it in data.get("stationboard", []):
        # Filter by direction to main station
        # if (it.get("to") or "") != target_destination:
        #    continue
        item = _format_departure_item(it, now_utc)
        if not item:
            continue
        out.append(item)
    return out

def fetch_mixed_departures(bus_station, train_station, limit, target_destination):
    # Fetch bus and train lists independently and then merge-sort by departure time
    bus = fetch_bus_departures(bus_station, limit)
    trains = fetch_train_departures_towards(train_station, limit, target_destination)
    merged = sorted(bus + trains, key=lambda e: e.get("_epoch", float("inf")))
    # Trim to limit and drop helper keys
    trimmed = []
    for e in merged[:limit]:
        e.pop("_epoch", None)
        trimmed.append(e)
    return trimmed

# -------------- rendering ----------------

def draw_frame(entries, tick, width=None, height=None):
    BLACK=(0,0,0); WHITE=(255,255,255); GREY=(170,170,170)
    DARKBG=(12,12,14); ORANGE=(245,160,20); RED=(220,60,60)

    # Use provided dimensions or fall back to default
    img_width = width or DISPLAY_WIDTH
    img_height = height or DISPLAY_HEIGHT
    
    img = Image.new("RGB", (img_width, img_height), BLACK)
    d = ImageDraw.Draw(img)

    # Header (Portrait)
    nowtxt = datetime.now().strftime("%H:%M:%S")
    title = f"Abfahrten – Solarstrasse"
    d.text((12, 12), title, font=FONT_BIG, fill=WHITE)
    tw, th = text_size(d, nowtxt, FONT_MED)
    d.text((img_width-12-tw, 14), nowtxt, font=FONT_MED, fill=GREY)

    # Column headers (Portrait)
    y = 50
    d.line((10, y, img_width-10, y), fill=GREY)
    d.text((14, 30), "Ab", font=FONT_MED, fill=GREY)
    d.text((70, 30), "Linie", font=FONT_MED, fill=GREY)
    d.text((120, 30), "Ziel", font=FONT_MED, fill=GREY)
    d.text((img_width-50, 30), "min", font=FONT_MED, fill=GREY)

    row_h = 36  # Increased row height to prevent text overlap
    y += 12
    for i, e in enumerate(entries[:LIMIT]):
        y += row_h
        if y + row_h > img_height - 10:  # Prevent overflow
            break
            
        if i % 2:
            d.rounded_rectangle((8, y-row_h+2, img_width-8, y+2), radius=6, fill=DARKBG)
        
        # Line badge
        lc = line_color(e["cat"], e["line"])
        d.rounded_rectangle((65, y-24, 115, y+2), radius=6, fill=lc)
        d.text((70, y-22), e["line"][:3], font=FONT_MED, fill=(0,0,0))
        
        # Time + delay color
        col = WHITE if e["delay"]<=0 else (ORANGE if e["delay"]<6 else RED)
        d.text((12, y-22), e["time"], font=FONT_BIG, fill=col)
        if e["delay"]>0:
            d.text((45, y-20), f"+{e['delay']}", font=FONT_MED, fill=col)
        
        # Destination (truncate)
        to = e["to"]
        maxw = img_width - 120 - 50
        while text_width(d, to, FONT_MED) > maxw and len(to) > 1:
            to = to[:-1]
        if text_width(d, to, FONT_MED) > maxw:
            to = to[:-1] + "…"
        d.text((120, y-22), to, font=FONT_MED, fill=WHITE)
        
        # Countdown
        if e["mins"] == 0:
            txt = "now" if tick else f"{e['secs']:02d}s"
        else:
            sep = ":" if tick else " "
            txt = f"{e['mins']}{sep}{e['secs']:02d}"
        d.text((img_width-48, y-22), txt, font=FONT_BIG, fill=WHITE)

    return img

# -------------- main ---------------------

def main():
    # Parse command line arguments
    max_frames = None
    if len(sys.argv) > 1 and sys.argv[1] == "--frames" and len(sys.argv) > 2:
        try:
            max_frames = int(sys.argv[2])
            print(f"Will generate maximum {max_frames} frames")
        except ValueError:
            print("Invalid frame count, ignoring --frames argument")
    
    # Check if framebuffer device exists
    try:
        fb = os.open(FRAMEBUFFER_DEVICE, os.O_RDWR)
        framebuffer_mode = True
        print(f"Running in framebuffer mode ({FRAMEBUFFER_DEVICE})")
    except FileNotFoundError:
        framebuffer_mode = False
        print(f"Framebuffer device {FRAMEBUFFER_DEVICE} not found - running in image output mode")
        
        # Determine output directory for images
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = script_dir  # Save images in the same directory as the script
        print(f"Images will be saved in: {output_dir}")
        print("Images will be saved as 'departure_board_*.png'")
        if max_frames:
            print(f"Will stop after {max_frames} frames")
    
    if framebuffer_mode:
        info = get_fbinfo(fb)
        xres, yres, bpp = info["xres"], info["yres"], info["bpp"]
        
        print(f"Framebuffer info: {xres}x{yres}, {bpp} bpp")
        print(f"Expected display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        
        if bpp != 16:
            print(f"Warning: bpp = {bpp} (expected 16/RGB565)")

        stride = get_stride_sysfs() or get_stride_default(xres)
        print(f"Stride: {stride} (expected: {xres * 2})")
        
        # Use native framebuffer dimensions; rotate if landscape to fill screen
        fb_is_landscape = xres >= yres
        if fb_is_landscape:
            # We'll render in rotated portrait (yres x xres) and rotate to fill fb
            actual_width, actual_height = yres, xres
            print(f"Framebuffer is landscape {xres}x{yres} → rendering {actual_width}x{actual_height} and rotating to fit")
        else:
            actual_width, actual_height = xres, yres
            print(f"Framebuffer is portrait {xres}x{yres} → rendering native size")
        
        size = stride * yres
        mm = mmap.mmap(fb, length=size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE, offset=0)

        # Clear the screen on startup to remove any previous content
        clear_framebuffer(mm, xres, yres, stride)
        time.sleep(0.1)  # Small delay to ensure clear operation completes
    else:
        # Image output mode - use display dimensions directly
        actual_width, actual_height = DISPLAY_WIDTH, DISPLAY_HEIGHT
        print(f"Using display dimensions: {actual_width}x{actual_height}")

    entries = []
    last = 0
    tick = True
    frame_count = 0
    
    try:
        while True:
            now = time.time()
            if now - last > REFRESH_SECS or not entries:
                try:
                    entries = fetch_mixed_departures(BUS_STATION, TRAIN_STATION, LIMIT, TARGET_DESTINATION)
                    print(f"Fetched {len(entries)} departures")
                except Exception as e:
                    print(f"API Error: {e}")
                    img = Image.new("RGB", (actual_width, actual_height), (0,0,0))
                    d = ImageDraw.Draw(img)
                    d.text((12, 12), f"API-Fehler: {e}", font=FONT_REG, fill=(220,60,60))
                    
                    if framebuffer_mode:
                        # Scale if needed to match framebuffer dimensions
                        if (xres, yres) != (actual_width, actual_height):
                            img = img.resize((xres, yres))
                        
                        out_bytes = rgb888_to_rgb565(img)
                        # Write to framebuffer with proper stride handling
                        for y in range(yres):
                            mm.seek(y * stride)
                            row_start = y * xres * 2
                            row_end = row_start + xres * 2
                            mm.write(out_bytes[row_start:row_end])
                    else:
                        # Save error image
                        error_filename = os.path.join(output_dir, f"departure_board_error_{frame_count:04d}.png")
                        img.save(error_filename)
                        print(f"Saved error image: {error_filename}")
                last = now

            # Draw frame using chosen render dimensions
            frame = draw_frame(entries, tick, actual_width, actual_height)

            # Optional: replace first frame with a clear test pattern to validate output
            if DRAW_TEST_PATTERN_ONCE and frame_count == 0:
                tp = Image.new("RGB", (actual_width, actual_height))
                td = ImageDraw.Draw(tp)
                # Vertical color bars and a diagonal line
                cols = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255)]
                bar_w = max(1, actual_width // len(cols))
                for i, c in enumerate(cols):
                    x0 = i * bar_w
                    td.rectangle((x0, 0, min(actual_width-1, x0 + bar_w - 1), actual_height-1), fill=c)
                td.line((0,0, actual_width-1, actual_height-1), fill=(0,0,0), width=5)
                td.text((10,10), "TEST", font=FONT_BIG, fill=(0,0,0))
                frame = tp
            
            if framebuffer_mode:
                # Rotate to match landscape framebuffer if needed
                if 'fb_is_landscape' in locals() and fb_is_landscape:
                    # Rotate so portrait render fills landscape fb
                    frame = frame.rotate(LANDSCAPE_ROTATE_DEG, expand=True)
                # Ensure exact fb size
                if frame.size != (xres, yres):
                    frame = frame.resize((xres, yres))
                
                out_bytes = rgb888_to_rgb565(frame)
                
                # Write to framebuffer with proper stride handling
                for y in range(yres):
                    mm.seek(y * stride)
                    row_start = y * xres * 2
                    row_end = row_start + xres * 2
                    mm.write(out_bytes[row_start:row_end])
                mm.flush()
            else:
                # Save image file
                filename = os.path.join(output_dir, f"departure_board_{frame_count:04d}.png")
                frame.save(filename)
                print(f"Saved: {filename}")
                frame_count += 1
                
                # Check if we've reached the maximum frame count
                if max_frames and frame_count >= max_frames:
                    print(f"Reached maximum frame count ({max_frames}), stopping")
                    break

            tick = not tick
            time.sleep(1)
    finally:
        if framebuffer_mode:
            mm.close()
            os.close(fb)

if __name__ == "__main__":
    main()
