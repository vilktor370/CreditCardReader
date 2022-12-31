# CreditCardReader

This project reads an credit card image and reads the card number of the card.

Example input:

![Credit card](ocr_input.jpg)

Output:

![Detect card](detection_result.png)

The idea is to use contour detection to find the contours of each digit. Then matching against a reference image:

![Reference image](ocr_a_reference_bitwise.jpg)

OpenCV and C++ is used.