// far field resolution
dx1 = 0.050;
// resolution at cylinder
dx2 = 0.025;

// 4 corners of rectangle
Point(1) = {0, 0, 0, dx1};
Point(2) = {2.2, 0, 0, dx1};
Point(3) = {0, 0.41, 0, dx1};
Point(4) = {2.2, 0.41, 0, dx1};

// define the rectangle
Line(1) = {3, 4};
Line(2) = {4, 2};
Line(3) = {2, 1};
Line(4) = {1, 3};
Line Loop(11) = {1, 2, 3, 4};
Plane Surface(11) = {11};  // Updated surface tag for the rectangle

// Define the circle (cutout)
radius = 0.05; // radius of the cutout
centerX = 0.2; // x-coordinate of the circle center
centerY = 0.2; // y-coordinate of the circle center

// Circle center and perimeter points
Point(5) = {centerX, centerY, 0, dx2};             // Center
Point(6) = {centerX + radius, centerY, 0, dx2};    // Point on the right (start of circle)
Point(7) = {centerX - radius, centerY, 0, dx2};    // Point on the left

// Create two half-circles
Circle(5) = {6, 5, 7};  // Top half
Circle(6) = {7, 5, 6};  // Bottom half

// Create a loop and surface for the circular cutout
Line Loop(12) = {5, 6}; // Full circle
Plane Surface(12) = {12}; // Updated surface tag for the circle

// Surface Loop for subtraction (rectangle - circle)
Surface Loop(13) = {11}; // Rectangle
Surface Loop(14) = {12}; // Circle (cutout)
Surface(15) = {11, -12}; // Create a surface where circle is subtracted from rectangle

// Physical groups for boundary conditions
// Top (1) and bottom (3) edges of the rectangle
Physical Line(1) = {1, 3};
// Inflow boundary (left)
Physical Line(2) = {4};
// Outflow boundary (right)
Physical Line(3) = {2};
// Boundary of the circular cutout
Physical Line(4) = {5, 6};

// Whole domain ID, including cutout
Physical Surface(15) = {15}; // Physical surface of the domain, after subtraction
