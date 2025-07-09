element_size_coarse = 0.04;

// Define corner points
Point(1) = {0, 1, 0, element_size_coarse};
Point(2) = {0, 0, 0, element_size_coarse};
Point(3) = {1, 0, 0, element_size_coarse};
Point(4) = {1, 0.5, 0, element_size_coarse};

Point(5) = {0.5, 0.5, 0, element_size_coarse};
Point(6) = {0.5, 1, 0, element_size_coarse};

// Define lines
Line(21) = {1, 2};
Line(22) = {2, 3};
Line(23) = {3, 4};
Line(24) = {4, 5};
Line(25) = {5, 6};
Line(26) = {6, 1};

// Define surface
Line Loop(215) = {21, 22, 23, 24, 25, 26};
Plane Surface(216) = {215};

// Physical groups
Physical Surface(17) = {216};
Physical Line(31) = {21};
Physical Line(32) = {22};
Physical Line(33) = {23};
Physical Line(34) = {24};
Physical Line(35) = {25};
Physical Line(36) = {26};
