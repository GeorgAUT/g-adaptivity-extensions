basin_x = 0.5;
basin_y = 0.3;

headland_x = 0.1;
headland_y = 0.1;

channel_width = 0.15; // Width of the channel in the middle

element_size_coarse = 0.02;

// Define corner points
Point(1) = {0, .25, 0, element_size_coarse};
Point(2) = {0, -0.25, 0, element_size_coarse};
Point(3) = {0.205, -0.25, 0, element_size_coarse};
Point(4) = {0.205, -0.1, 0, element_size_coarse};

Point(5) = {0.295, -0.1, 0, element_size_coarse};
Point(6) = {0.295, -0.25, 0, element_size_coarse};
Point(7) = {0.5, -0.25, 0, element_size_coarse};
Point(8) = {0.5, 0.25, 0, element_size_coarse};

Point(9) = {0.295, 0.25, 0, element_size_coarse};
Point(10) = {0.295, 0.1, 0, element_size_coarse};
Point(11) = {0.205, 0.1, 0, element_size_coarse};
Point(12) = {0.205, 0.25, 0, element_size_coarse};

// Define lines
Line(21) = {1, 2};
Line(22) = {2, 3};
Line(23) = {3, 4};
Line(24) = {4, 5};
Line(25) = {5, 6};
Line(26) = {6, 7};
Line(27) = {7, 8};
Line(28) = {8, 9};
Line(29) = {9, 10};
Line(210) = {10, 11};
Line(211) = {11, 12};
Line(212) = {12, 1};

// Define surface
Line Loop(215) = {21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212};
Plane Surface(216) = {215};

// Physical groups
Physical Surface(17) = {216};
Physical Line(31) = {21};
Physical Line(32) = {22};
Physical Line(33) = {23};
Physical Line(34) = {24};
Physical Line(35) = {25};
Physical Line(36) = {26};
Physical Line(37) = {27};
Physical Line(38) = {28};
Physical Line(39) = {29};
Physical Line(310) = {210};
Physical Line(311) = {211};
Physical Line(312) = {212};
