#include  "gpu_kernels.h"
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    std::vector<Point> gr = {}, n_gr = {};
    const std::vector<Point> h_points = { {0,0,1}, {1,0,2} , {0,1,1}, {1,1,2}, {0.5,0.5,1.5}, {0,0,2} };
    estimateGroundCUDA(h_points, gr, n_gr, 3, 3, 0.5);

    for (size_t i = 0; i < gr.size(); i++)
    {
        std::cout<< "Ground Point " << i << ": " << gr[i].x << ", " << gr[i].y << ", " << gr[i].z << std::endl;
    }
    
    for (size_t i = 0; i < n_gr.size(); i++)
    {
        std::cout<< "Non-Ground Point " << i << ": " << n_gr[i].x << ", " << n_gr[i].y << ", " << n_gr[i].z << std::endl;
    }
    
    return 0;
}
