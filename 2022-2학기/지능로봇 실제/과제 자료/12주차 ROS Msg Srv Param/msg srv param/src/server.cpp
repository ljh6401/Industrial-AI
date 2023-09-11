#include <ros/ros.h>
#include <assign/msg.h>
#include <assign/srv.h>
#include <cstdlib>

int iter;

bool calc_area(assign::srv::Request &req, assign::srv::Response &res);


int main(int argc, char **argv)
{
    iter = 0;

    ros::init(argc, argv, "Server");
    ros::NodeHandle n;
    ros::ServiceServer service = n.advertiseService("calc_area", calc_area); // Service Request �� calc_area �Լ� ����

    ROS_INFO("Server");
    ros::spin();

    return 0;
}


bool calc_area(assign::srv::Request &req, assign::srv::Response &res){

    ROS_INFO("Sub Radius %d", iter);

    std::vector<float> Area; // Area�� ������ array Area ����

    for(int i = 0; i< 5; i++) {
        ROS_INFO("%d", req.radius[i]);                        // request input���� ���� Radius �� ���

        Area.push_back(req.radius[i] * req.radius[i] * 3.14); // Area ��� �� array Area�� push_back
    }

    res.area = Area;

    iter += 1;

    return true; // true ��ȯ
}
