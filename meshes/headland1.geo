basin_x = .5;
basin_y = .25;

headland_x_scale = 0.2;
headland_y = 0.1;

element_size_coarse = 0.02;

// Generate top headland
res = 400;
For k In {0:res:1}
    x = basin_x/res*k;
    b = 0.01;
    y = basin_y - headland_y*Exp(-0.5*((headland_x_scale*(x-basin_x/2))/b)^2);
    Point(10+k) = {x, y, 0, element_size_coarse*(y-(basin_y-headland_y))/headland_y + 0.1*element_size_coarse*(basin_y-y)/headland_y};
EndFor

// Generate bottom headland (mirrored version)
For k In {0:res:1}
    x = basin_x/res*k;
    b = 0.01;
    y = -basin_y + headland_y*Exp(-0.5*((headland_x_scale*(x-basin_x/2))/b)^2);
    Point(500+k) = {x, y, 0, element_size_coarse*(-y-(basin_y-headland_y))/headland_y + 0.1*element_size_coarse*(basin_y+y)/headland_y};
EndFor

// Generate headland curves
BSpline(100) = {10 : res+10}; // Top headland
BSpline(102) = {500 : res+500}; // Bottom headland

// Connect lines
Line(101) = {10, 500};
Line(103) = {res+500, res+10};

// Create loops for surfaces
Line Loop(104) = {100, -103, -102, -101};
Plane Surface(111) = {104};

// Physical definitions

Physical Surface(112) = {111};
Physical Line(1) = {101};
Physical Line(2) = {103};
Physical Line(3) = {102};
Physical Line(4) = {100};
