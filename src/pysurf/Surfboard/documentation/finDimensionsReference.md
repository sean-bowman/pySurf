# Surfboard Fin Dimensions Reference

Reference data used for parametric fin geometry generation in the
Surfboard domain. Dimensions sourced from FCS and Futures
fin manufacturers.

Sean Bowman [02/03/2026]

---

## Key Measurements

- **Base**: Length between leading and trailing edge where the fin meets the board. Linked to drive.
- **Depth (Height)**: Distance the fin penetrates into the water. Directly relates to hold.
- **Area**: Total fin surface area. More area = more hold and stability.
- **Sweep (Rake)**: Angle measuring how far the fin outline curves backward. More sweep = longer turning arc.
- **Cant**: Outward tilt angle of side fins from vertical. More cant = more responsiveness.
- **Foil**: Cross-sectional shape of the fin (flat, 80/20, symmetric/50-50).

---

## FCS II Fin Specifications

### Thruster (Performer Series, Medium)

| Dimension | Center | Side |
| --------- | ------ | ---- |
| Base | 4.34" / 110mm | 4.34" / 110mm |
| Depth | 4.57" / 116mm | 4.57" / 116mm |
| Area | 14.96 in^2 / 9,650mm^2 | 14.96 in^2 / 9,650mm^2 |
| Sweep | 33.7 deg | 33.7 deg |
| Foil | IFT | IFT |

### Reactor Series (Medium)

| Dimension | Value |
| --------- | ----- |
| Base | 4.34" / 110mm |
| Depth | 4.57" / 116mm |
| Area | 14.96 in^2 / 9,650mm^2 |
| Sweep | 31.9 deg |
| Foil | Flat |

### V2 PC Tri-Quad (Medium)

| Dimension | Side (Tri/Quad Front) | Quad Rear |
| --------- | --------------------- | --------- |
| Base | 4.33" / 110mm | 4.10" / 104mm |
| Depth | 4.53" / 115mm | 4.26" / 108mm |
| Area | 15.23 in^2 | 12.64 in^2 |
| Sweep | 32.0 deg | 32.4 deg |
| Foil | Flat | 80/20 |

### Single Fins

| Size | Base | Depth | Area | Sweep | Foil |
| ---- | ---- | ----- | ---- | ----- | ---- |
| 6" | 4.89" / 124mm | 6.00" / 152mm | 21.19 in^2 / 13,672mm^2 | 32.7 deg | 50/50 |
| 7" | 5.70" / 145mm | 7.01" / 178mm | 28.84 in^2 / 18,607mm^2 | 32.7 deg | 50/50 |
| 8" | 6.51" / 165mm | 8.01" / 203mm | 37.67 in^2 / 24,304mm^2 | 32.7 deg | 50/50 |

---

## Futures Fins Specifications

### F6 Legacy Quad (Medium, 125-175 lbs)

| Dimension | Front (Side) | Rear (Trailer) |
| --------- | ------------ | -------------- |
| Base | 4.35" / 110mm | 3.87" / 98mm |
| Depth | 4.56" / 116mm | 4.05" / 103mm |
| Area | 15.12 in^2 | 11.67 in^2 |
| Cant Angle | 6.5 deg | 3.0 deg |
| Foil | Flat | 80/20 |

### Tyler Warren Quad (True Ames / Futures Compatible)

| Dimension | Front | Rear |
| --------- | ----- | ---- |
| Base | 4.54" / 115mm | 3.67" / 93mm |
| Depth | 4.69" / 119mm | 3.62" / 92mm |
| Area | 15.96 in^2 / 103 cm^2 | 10.70 in^2 / 67 cm^2 |

### EN Twin (X-Large)

| Dimension | Side Fin |
| --------- | -------- |
| Base | 5.44" / 138mm |
| Depth | 5.55" / 141mm |
| Area | 22.75 in^2 |
| Foil | Flat |

### Christenson Keel Twin

| Dimension | Side Fin |
| --------- | -------- |
| Base | 6.54" / 166mm |
| Depth | 5.04" / 128mm |
| Area | 22.47 in^2 |
| Cant Angle | 4.0 deg |
| Foil | Flat |

---

## Dimensions Used in the Parametric Generator

The following values were chosen for our parametric generator, drawing from
the industry reference data above. All values are in mm and scale linearly
with board length using `scaleFactor = boardLength / 1828mm` (normalized
to a 6'0" shortboard).

### Thruster (3 fins)

| Parameter | Center Fin | Side Fins |
| --------- | ---------- | --------- |
| Height | 115mm (~4.5") | 105mm (~4.1") |
| Base Chord | 110mm (~4.3") | 100mm (~3.9") |
| Thickness | 8mm | 8mm |
| Cant Angle | 0 deg | 5 deg |
| Trailing Margin | 20mm | 180mm |
| Lateral Offset | 0mm (center) | 100mm |

### Twin (2 fins)

| Parameter | Side Fins |
| --------- | --------- |
| Height | 115mm (~4.5") |
| Base Chord | 110mm (~4.3") |
| Thickness | 8mm |
| Cant Angle | 7 deg |
| Trailing Margin | 220mm |
| Lateral Offset | 120mm |

Rationale: Larger than thruster sides to compensate for missing center fin.
Higher cant angle for more drive. Positioned wider and further forward for
a loose, skatey feel. Referenced FCS II Mark Richards Twin and Futures EN Twin.

### Quad (4 fins)

| Parameter | Front Pair | Rear Pair (Trailers) |
| --------- | ---------- | -------------------- |
| Height | 105mm (~4.1") | 75mm (~3.0") |
| Base Chord | 100mm (~3.9") | 70mm (~2.75") |
| Thickness | 8mm | 8mm |
| Cant Angle | 5 deg | 3 deg |
| Trailing Margin | 180mm | 50mm |
| Lateral Offset | 100mm | 115mm |

Rationale: Front fins match thruster sides. Rear trailer fins are ~71%
height and ~70% base of fronts, consistent with FCS V2 Quad and Futures F6
Legacy ratios. Rear fins have less cant for tracking stability and are
positioned slightly wider to channel water between the pairs.

### Single (1 fin)

| Parameter | Center Fin |
| --------- | ---------- |
| Height | 175mm (~6.9") |
| Base Chord | 165mm (~6.5") |
| Thickness | 12mm |
| Cant Angle | 0 deg |
| Trailing Margin | 80mm |
| Lateral Offset | 0mm (center) |

Rationale: Larger and thicker than thruster center fin since it provides
all directional stability alone. Dimensions between FCS 7" and 8" single
fin specs. Scales appropriately for longboards (at 9'0", becomes ~262mm
height / ~10.3").

---

## Sources

- [FCS Fin Data](https://www.surffcs.com/pages/fcs-fin-data)
- [FCS Fin Data (Central Coast Surfboards mirror)](https://ccsurf.com/pages/fcs-fin-data)
- [A Guide to FCS Fins](https://www.surffcs.com/blogs/community/a-guide-to-fcs-fins)
- [Futures Fins F6 Legacy Quad](https://surftech.com/products/f6-legacy-quad-fins)
- [Tyler Warren Quad (True Ames)](https://www.trueames.com/products/tyler-warren-quad-solid-fiberglass-futures-compatible)
- [Futures Performance Single Fin](https://www.cleanlinesurf.com/products/futures-fins-performance-single-fin)
