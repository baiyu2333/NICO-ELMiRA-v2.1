"""Vector checkerboard for printing. Nominal size is not measured calibration."""
from pathlib import Path

from .common import positive, write_json


def generate_board(output, square_mm=25):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas

    square_mm = positive(square_mm, "square_mm")
    if not 10 <= square_mm <= 25:
        raise ValueError("A4 board square size must be 10..25 mm")
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError("Board output already exists; use a new filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = landscape(A4)
    square = square_mm*mm
    x0, y0 = (width-10*square)/2, (height-7*square)/2
    canvas = Canvas(str(output), pagesize=(width, height), pageCompression=0, invariant=1)
    canvas.setTitle("NICO single-RGB calibration: 9 x 6 inner corners")
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColorRGB(0, 0, 0)
    for row in range(7):
        for column in range(10):
            if (row+column) % 2 == 0:
                canvas.rect(x0+column*square, y0+(6-row)*square, square, square, stroke=0, fill=1)
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(width/2, height-8*mm,
                             "NICO calibration | 9 x 6 INNER CORNERS | 10 x 7 squares | nominal %g mm/square" % square_mm)
    canvas.setFont("Helvetica", 6)
    canvas.drawString(x0, y0+7*square+2*mm, "TOP LEFT (orientation reference; not an automatically detected marker)")
    canvas.setLineWidth(.5)
    canvas.line(x0, 10*mm, x0+100*mm, 10*mm)
    for offset in range(0, 101, 10):
        tick = 2 if offset in (0, 100) else 1
        canvas.line(x0+offset*mm, (10-tick)*mm, x0+offset*mm, (10+tick)*mm)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(x0+105*mm, 9*mm, "100 mm ruler: measure between end ticks")
    canvas.drawCentredString(width/2, 4.5*mm, "Print A4 LANDSCAPE at ACTUAL SIZE / 100%. No Fit or Shrink. Measure printed squares before calibration.")
    canvas.showPage()
    canvas.save()
    record = {"pdf": str(output.resolve()), "paper": "A4 landscape", "page_size_mm": [297, 210],
              "squares": [10, 7], "inner_corners": [9, 6], "nominal_square_mm": square_mm,
              "grid_size_mm": [10*square_mm, 7*square_mm], "scale_ruler_mm": 100,
              "measured_square_mm": None, "print_verified": False,
              "camera_calibration_verified": False, "real_robot_motion_commanded": False}
    write_json(output.with_suffix('.json'), record)
    return record
