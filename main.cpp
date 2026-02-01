#include  "gpu_kernels.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class ControllerNode : public rclcpp::Node
{
public:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cp_pub;
    ControllerNode() : Node("controller_node")
    {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);

        this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_points", qos,
                                                                                std::bind(&ControllerNode::scanCallback, this, std::placeholders::_1));
        this->segmented_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_pc", 100);
    }


    void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
    {
        /* CONVERSION */
        std::vector<Point> h_points;

        for (std::uint32_t r = 0; r < sub_cloud->height; ++r)
        {
            const std::uint8_t *row_data = sub_cloud->data.data() + r * sub_cloud->row_step;

            for (std::uint32_t c = 0; c < sub_cloud->width; ++c)
            {
                const std::uint8_t *pt_data = row_data + c * sub_cloud->point_step;

                Point p;
                memcpy(&p, pt_data, sizeof(Point));
                h_points.push_back(p);
            }
        }

        std::vector<Point> gr = {}, n_gr = {};

        estimateGroundCUDA(h_points, gr, n_gr, 3, 20, 0.15);

        /* PUBLISHING */
        auto segmented_pc_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        segmented_pc_msg->header = sub_cloud->header;
        segmented_pc_msg->height = 1;
        segmented_pc_msg->width = n_gr.size();
        segmented_pc_msg->is_dense = false;
        segmented_pc_msg->is_bigendian = false;
        segmented_pc_msg->point_step = sizeof(Point);
        segmented_pc_msg->row_step = segmented_pc_msg->point_step * segmented_pc_msg->width;
        segmented_pc_msg->data.resize(segmented_pc_msg->row_step * segmented_pc_msg->height);
        segmented_pc_msg->fields.resize(3);
        segmented_pc_msg->fields[0].name = "x";
        segmented_pc_msg->fields[0].offset = 0;
        segmented_pc_msg->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        segmented_pc_msg->fields[0].count = 1;
        segmented_pc_msg->fields[1].name = "y";
        segmented_pc_msg->fields[1].offset = 4;
        segmented_pc_msg->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        segmented_pc_msg->fields[1].count = 1;
        segmented_pc_msg->fields[2].name = "z";
        segmented_pc_msg->fields[2].offset = 8;
        segmented_pc_msg->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        segmented_pc_msg->fields[2].count = 1;
        for (size_t i = 0; i < n_gr.size(); i++)
        {
            std::uint8_t *pt_data = segmented_pc_msg->data.data() + i * segmented_pc_msg->point_step;
            memcpy(pt_data, &n_gr[i], sizeof(Point));
        }
        this->segmented_cp_pub->publish(*segmented_pc_msg);
    }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<ControllerNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
