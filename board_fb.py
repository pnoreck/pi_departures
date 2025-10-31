#!/usr/bin/env python3
import os, time, mmap, fcntl, struct, requests, sys
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

# ---------------- Config ----------------
# City bus stop in front of the house
BUS_STATION = "Winterthur, Else Züblin"

# Train station behind the house
TRAIN_STATION = "Winterthur Hegi"

# We only care about trains heading toward the main station
FILTER_BUS = "Elsau, Melcher"
FILTER_TRAIN = "Wil SG"

# Debugging for train request
DEBUG_TRAINS = False
DEBUG_TRAIN_LIMIT_PRINT = 5

# Display dimensions - Portrait orientation (320x480)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 320, 480  # Portrait orientation

# Number of departures to fetch and display
LIMIT = 8

# Refresh interval in seconds
REFRESH_SECS = 30

# Framebuffer device to use
FRAMEBUFFER_DEVICE = "/dev/fb1"  # Change to /dev/fb0, /dev/fb1, etc. as needed

# Font setup with crisp candidates and environment override
def _load_font(size):
    env_font = os.getenv("BOARD_FONT")
    candidates = [
        env_font,
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            font = ImageFont.truetype(path, size)
            return font, path
        except Exception:
            continue
    # Fallback to PIL default bitmap font
    try:
        return ImageFont.load_default(), "PIL:default"
    except Exception:
        # Last resort: re-raise to fail fast
        raise

FONT_REG, FONT_REG_NAME = _load_font(12)
FONT_MED, FONT_MED_NAME = _load_font(15)
FONT_BIG, FONT_BIG_NAME = _load_font(18)

# No rotation needed - we'll work directly in portrait mode
ROTATE_IF_LANDSCAPE = 0

# Framebuffer color format - try different combinations to get correct colors:
# BGR565: True = BGR565 (B-G-R), False = RGB565 (R-G-B)
# USE_BYTESWAP: True = swap bytes (big-endian), False = native (little-endian)
# Test combinations:
#   1. BGR565=True, USE_BYTESWAP=True  (original - was light blue/violet)
#   2. BGR565=False, USE_BYTESWAP=False (was light brown)
#   3. BGR565=False, USE_BYTESWAP=True  (testing now)
#   4. BGR565=True, USE_BYTESWAP=False (try if 3 doesn't work)
BGR565 = True
USE_BYTESWAP = True  # Try with byteswap - we're close with BGR565

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
    # Read color format offsets and lengths (at offsets 36, 40, 44, 48, 52, 56, 60, 64)
    red_offset = struct.unpack_from("<I", buf, 36)[0]
    red_length = struct.unpack_from("<I", buf, 40)[0]
    green_offset = struct.unpack_from("<I", buf, 44)[0]
    green_length = struct.unpack_from("<I", buf, 48)[0]
    blue_offset = struct.unpack_from("<I", buf, 52)[0]
    blue_length = struct.unpack_from("<I", buf, 56)[0]
    return {
        "xres": xres, "yres": yres, "bpp": bpp,
        "red_offset": red_offset, "red_length": red_length,
        "green_offset": green_offset, "green_length": green_length,
        "blue_offset": blue_offset, "blue_length": blue_length,
    }

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

def draw_text_solid(img: Image.Image, position, text: str, font: ImageFont.FreeTypeFont, fill):
    """Draw text as solid-colored glyphs using a monochrome mask to avoid color fringing on RGB565.
    
    Disables anti-aliasing by rendering at high resolution, thresholding to pure black/white,
    then scaling down with nearest neighbor for maximum sharpness on RGB565 displays.
    """
    # Measure text to size mask tightly
    tmp_draw = ImageDraw.Draw(img)
    l, t, r, b = tmp_draw.textbbox((0, 0), text, font=font)
    w, h = max(1, r - l), max(1, b - t)
    w, h = int(w), int(h)
    
    # Render at 3x resolution to get better glyph shape, then threshold to remove anti-aliasing
    scale = 3
    w_hires, h_hires = w * scale, h * scale
    mask_hires = Image.new("L", (w_hires, h_hires), 0)
    md_hires = ImageDraw.Draw(mask_hires)
    
    # Create high-resolution font
    try:
        if hasattr(font, 'path') and font.path:
            font_hires = ImageFont.truetype(font.path, int(font.size * scale))
        else:
            font_hires = font
    except:
        font_hires = font
    
    # Render text at high resolution
    text_x_hires, text_y_hires = int((-l) * scale), int((-t) * scale)
    md_hires.text((text_x_hires, text_y_hires), text, font=font_hires, fill=255)
    
    # Threshold to pure black/white (removes anti-aliasing gray pixels)
    # Any pixel >= 128 becomes 255 (fully opaque), < 128 becomes 0 (fully transparent)
    import numpy as np
    mask_array = np.asarray(mask_hires, dtype=np.uint8)
    mask_array = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    mask_hires = Image.fromarray(mask_array, mode='L')
    
    # Scale down using nearest neighbor to preserve sharp edges (no blur)
    mask = mask_hires.resize((w, h), resample=Image.NEAREST)
    
    x, y = position
    x, y = int(x), int(y)
    # Paste solid color through mask for crisp edges without colored halos
    img.paste(fill, (x, y, x + w, y + h), mask)

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
    # Framebuffer byte order - configurable via USE_BYTESWAP
    if USE_BYTESWAP:
        return rgb565.byteswap().tobytes()  # Big-endian
    else:
        return rgb565.astype('<u2').tobytes()  # Little-endian (native)

def line_color(cat, name):
    # Adobe Color palette: #025259, #007172, #F29325, #D94F04, #F4E2DE
    DARK_TEAL=(2,82,89)       # #025259
    TEAL=(0,113,114)          # #007172
    ORANGE=(242,147,37)       # #F29325
    RED_ORANGE=(217,79,4)     # #D94F04
    LIGHT_BEIGE=(244,226,222) # #F4E2DE
    
    c = (cat or "").upper(); n = (name or "").upper()

    # S-Bahn - use orange
    if c.startswith("S") or n.startswith("S"): return ORANGE

    # Tram - use teal
    if c in ("TRAM", "T"): return TEAL

    # Bus/Nachtbus - use red-orange
    if c in ("BUS","N","SN") or (n and n[0].isdigit()):
        return RED_ORANGE

    # Fernverkehr - use light beige
    if c in ("IC","IR","RE","RJ","RJX","EN"): return LIGHT_BEIGE
    
    # Default - use dark teal
    return DARK_TEAL

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

def fetch_departures(station, limit):
    url = "https://transport.opendata.ch/v1/stationboard"
    params = {"station": station, "limit": limit, "transportations[]": ["bus"]}

    # If this is the train station call, request all modes (remove bus filter) and debug-print URL
    if station == TRAIN_STATION:
        params = {"station": station, "limit": limit}
        if DEBUG_TRAINS:
            try:
                prepped = requests.Request("GET", url, params=params).prepare()
                print(f"[DEBUG] Train request URL: {prepped.url}")
            except Exception as _e:
                print(f"[DEBUG] Could not prepare train URL: {_e}")

    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    now_utc = datetime.now(timezone.utc)
    out = []
    for it in data.get("stationboard", []):
        item = _format_departure_item(it, now_utc)
        if not item:
            continue
        # Direction filters per station (case-insensitive substring match)
        if station == BUS_STATION and FILTER_BUS:
            try:
                if FILTER_BUS.lower() in (item.get("to") or "").lower():
                    continue
            except Exception:
                pass
        if station == TRAIN_STATION and FILTER_TRAIN:
            try:
                if FILTER_TRAIN.lower() in (item.get("to") or "").lower():
                    continue
            except Exception:
                pass
        out.append(item)

    # Extra train-side debug: summarize what we got from the train station
    if station == TRAIN_STATION and DEBUG_TRAINS:
        print(f"[DEBUG] Train results from '{station}': {len(out)} departures")
        for e in out[:DEBUG_TRAIN_LIMIT_PRINT]:
            print(f"[DEBUG]  {e['time']}  {e['cat']:<3} {e['line']:<6} → {e['to']}  delay:+{e['delay']}")
    return out

def fetch_mixed_departures(bus_station, train_station, limit):
    # Fetch bus and train lists independently and then merge-sort by departure time
    bus = fetch_departures(bus_station, limit)
    trains = fetch_departures(train_station, limit)
    if DEBUG_TRAINS:
        print(f"[DEBUG] Buses fetched: {len(bus)}  from '{bus_station}'")
        print(f"[DEBUG] Trains fetched: {len(trains)} from '{train_station}'")
    merged = sorted(trains + bus, key=lambda e: e.get("_epoch", float("inf")))
    # Trim to limit and drop helper keys
    trimmed = []
    for e in merged[:limit]:
        e.pop("_epoch", None)
        trimmed.append(e)
    return trimmed

# -------------- rendering ----------------

def draw_frame(entries, tick, width=None, height=None):
    # Adobe Color palette
    # #025259, #007172, #F29325, #D94F04, #F4E2DE
    DARK_TEAL=(2,82,89)      # #025259 - main background
    TEAL=(0,113,114)         # #007172 - accents/secondary
    ORANGE=(242,147,37)      # #F29325 - warnings/delays
    RED_ORANGE=(217,79,4)    # #D94F04 - severe delays
    LIGHT_BEIGE=(244,226,222) # #F4E2DE - text and highlights
    
    # Map to semantic names
    BLACK=DARK_TEAL           # Background
    WHITE=LIGHT_BEIGE         # Primary text
    GREY=TEAL                 # Secondary text
    DARKBG=TEAL               # Alternating rows (darker)
    ORANGE_COLOR=ORANGE       # Delays < 6 min
    RED=RED_ORANGE            # Delays >= 6 min

    # Use provided dimensions or fall back to default
    img_width = width or DISPLAY_WIDTH
    img_height = height or DISPLAY_HEIGHT
    
    img = Image.new("RGB", (img_width, img_height), BLACK)
    d = ImageDraw.Draw(img)

    # Header (Portrait)
    nowtxt = datetime.now().strftime("%H:%M:%S")
    title = f"Abfahrten – Solarstrasse"
    draw_text_solid(img, (12, 12), title, FONT_BIG, WHITE)
    tw, th = text_size(d, nowtxt, FONT_MED)
    draw_text_solid(img, (img_width-12-tw, 14), nowtxt, FONT_MED, GREY)

    # Column headers (Portrait)
    y = 50
    d.line((10, y, img_width-10, y), fill=GREY)
    draw_text_solid(img, (14, 30), "Ab", FONT_MED, GREY)
    draw_text_solid(img, (70, 30), "Linie", FONT_MED, GREY)
    draw_text_solid(img, (120, 30), "Ziel", FONT_MED, GREY)
    draw_text_solid(img, (img_width-50, 30), "min", FONT_MED, GREY)

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
        # d.text((70, y-22), e["line"][:3], font=FONT_MED, fill=(0,0,0))
        
        # Time + delay color
        col = WHITE if e["delay"]<=0 else (ORANGE_COLOR if e["delay"]<6 else RED)
        draw_text_solid(img, (12, y-22), e["time"], FONT_BIG, col)
        if e["delay"]>0:
            draw_text_solid(img, (70, y-20), f"+{e['delay']}", FONT_MED, col)
        
        # Destination (truncate)
        to = e["to"]
        maxw = img_width - 120 - 50
        while text_width(d, to, FONT_MED) > maxw and len(to) > 1:
            to = to[:-1]
        if text_width(d, to, FONT_MED) > maxw:
            to = to[:-1] + "…"
        draw_text_solid(img, (120, y-22), to, FONT_MED, WHITE)
        
        # Countdown
        if e["mins"] == 0:
            txt = "now" if tick else f"{e['secs']:02d}s"
        else:
            sep = ":" if tick else " "
            txt = f"{e['mins']}{sep}{e['secs']:02d}"
        draw_text_solid(img, (img_width-50, y-22), txt, FONT_BIG, WHITE)

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
        # Declare global variables at the start before any use
        global BGR565, USE_BYTESWAP
        
        info = get_fbinfo(fb)
        xres, yres, bpp = info["xres"], info["yres"], info["bpp"]
        
        print(f"Framebuffer info: {xres}x{yres}, {bpp} bpp")
        print(f"Expected display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        
        # Print color format info to help diagnose
        red_off = info.get("red_offset", 0)
        red_len = info.get("red_length", 0)
        green_off = info.get("green_offset", 0)
        green_len = info.get("green_length", 0)
        blue_off = info.get("blue_offset", 0)
        blue_len = info.get("blue_length", 0)
        print(f"Color format: R offset={red_off} len={red_len}, G offset={green_off} len={green_len}, B offset={blue_off} len={blue_len}")
        
        # Auto-detect RGB565 vs BGR565 based on offsets
        # RGB565: R at bit 11, G at bit 5, B at bit 0
        # BGR565: B at bit 11, G at bit 5, R at bit 0
        auto_bgr565 = None
        if red_off == 11 and blue_off == 0:
            auto_bgr565 = False
            print("Detected RGB565 format (R=11, G=5, B=0)")
        elif blue_off == 11 and red_off == 0:
            auto_bgr565 = True
            print("Detected BGR565 format (B=11, G=5, R=0)")
        else:
            print(f"Unknown format (R={red_off}, G={green_off}, B={blue_off}) - using manual BGR565={BGR565}")
        
        # Use auto-detected format if available, otherwise use manual setting
        # NOTE: Auto-detection may fail due to structure layout - using tested values
        # From testing: BGR565=True, USE_BYTESWAP=False gave light green (closest to dark teal)
        if auto_bgr565 is not None and red_len > 0 and blue_len > 0:
            print(f"Auto-detected format: Using BGR565={auto_bgr565}")
            BGR565 = auto_bgr565
            USE_BYTESWAP = False  # Based on testing, this gives best results
            print(f"Using byte swap: {USE_BYTESWAP}")
        else:
            # Use tested values that work best
            print("Auto-detection unreliable, using tested values: BGR565=True, USE_BYTESWAP=False")
            BGR565 = True
            USE_BYTESWAP = False  # This gave light green (closest to dark teal)
        
        stride = get_stride_sysfs() or get_stride_default(xres)
        print(f"Stride: {stride} (expected: {xres * 2})")
        
        # Validate bpp vs stride - if stride suggests 16 bpp, trust that over reported bpp
        inferred_bpp = (stride // xres) * 8
        if bpp != 16:
            print(f"Warning: bpp = {bpp} (expected 16/RGB565)")
            if inferred_bpp == 16:
                print(f"Note: Stride indicates {inferred_bpp} bpp - will use 16 bpp RGB565 format")
                bpp = 16  # Override with stride-based detection
            else:
                print(f"Warning: Inferred bpp from stride is {inferred_bpp}, mismatch may cause display issues")
        print(f"Fonts: REG='{FONT_REG_NAME}', MED='{FONT_MED_NAME}', BIG='{FONT_BIG_NAME}'")
        
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
        print(f"Fonts: REG='{FONT_REG_NAME}', MED='{FONT_MED_NAME}', BIG='{FONT_BIG_NAME}'")

    entries = []
    last = 0
    tick = True
    frame_count = 0
    
    try:
        while True:
            now = time.time()
            if now - last > REFRESH_SECS or not entries:
                try:
                    entries = fetch_mixed_departures(BUS_STATION, TRAIN_STATION, LIMIT)
                    print(f"Fetched {len(entries)} departures")
                except Exception as e:
                    print(f"API Error: {e}")
                    img = Image.new("RGB", (actual_width, actual_height), (0,0,0))
                    d = ImageDraw.Draw(img)
                    draw_text_solid(img, (12, 12), f"API-Fehler: {e}", FONT_REG, (220,60,60))
                    
                    if framebuffer_mode:
                        # Scale if needed to match framebuffer dimensions
                        if (xres, yres) != (actual_width, actual_height):
                            img = img.resize((xres, yres), resample=Image.LANCZOS)
                        
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
                draw_text_solid(tp, (10,10), "TEST", FONT_BIG, (0,0,0))
                frame = tp
            
            if framebuffer_mode:
                # Rotate to match landscape framebuffer if needed
                if 'fb_is_landscape' in locals() and fb_is_landscape:
                    # Rotate so portrait render fills landscape fb - use LANCZOS for better quality
                    frame = frame.rotate(LANDSCAPE_ROTATE_DEG, expand=False, resample=Image.LANCZOS)
                # Ensure exact fb size - use LANCZOS for scaling, NEAREST only if exact size match
                if frame.size != (xres, yres):
                    print(f"Image needed to be resized: {xres}x{yres}")
                    frame = frame.resize((xres, yres), resample=Image.LANCZOS)
                
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
