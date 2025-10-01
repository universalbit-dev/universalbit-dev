# Fibonacci Clock Display Documentation

This documentation explains how to read the time from the Fibonacci Clock shown below, using the color and value legend.

![Fibonacci Smartwatch Clock](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/images/fibonacci_smartwatch.png)

---

## How to Read the Fibonacci Clock

The clock face is divided into five squares, each labeled with a Fibonacci number (1, 1, 2, 3, 5).  
Each square is lit with a color representing either "Hour", "Minute", or "Both".

| Square Value    | Color         | Hour Contribution | Minute Contribution |
|-----------------|--------------|-------------------|--------------------|
| 5               | Red (Hour)   | +5                | 0                  |
| 3               | Blue (Both)  | +3                | +3                 |
| 2               | Green (Minute)| 0                | +2                 |
| 1 (Center)      | Blue (Both)  | +1                | +1                 |
| 1 (Bottom-Right)| Green (Minute)| 0                | +1                 |

### Color Legend

- **Red**   = Hour only
- **Green** = Minute only
- **Blue**  = Both hour and minute

---

## Calculating the Time

1. **Sum the Red and Blue squares for Hours**
    - Red squares contribute their value to the hour sum.
    - Blue squares contribute their value to both hour and minute sums.

2. **Sum the Green and Blue squares for Minutes**
    - Green squares contribute their value to the minute sum.
    - Blue squares contribute their value to both hour and minute sums.

3. **Multiply the minute sum by 5**
    - The minute value is always rounded to the nearest 5 and calculated as:  
      **minute sum × 5**

---

### Example (as shown in the image above)

| Square        | Value | Color | Hour Sum | Minute Sum |
|---------------|-------|-------|----------|------------|
| Top-left      | 5     | Red   | +5       | 0          |
| Top-right     | 3     | Blue  | +3       | +3         |
| Bottom-left   | 2     | Green | 0        | +2         |
| Center        | 1     | Blue  | +1       | +1         |
| Bottom-right  | 1     | Green | 0        | +1         |

**Totals:**
- Hour sum   = 5 + 3 + 1 = **9**
- Minute sum = 3 + 2 + 1 + 1 = **7**

**Final Time:**  
- **Hour:** 9  
- **Minute:** 7 × 5 = **35**  
- **Displayed Time:** **9:35**

---

## Notes

- Only the colors matter for determining the time; ignore any squares that are not lit.
- The minute value will always be a multiple of 5.
- This method makes the clock both functional and a puzzle to read!

---

## Reference

This documentation covers the visual logic for the Fibonacci Clock.  
For more details about the electronic build and [code](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/fibonacci_clock_esp.ino), see the [main](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/ESP8266%20Fibonacci%20Clock.md) project documentation.
