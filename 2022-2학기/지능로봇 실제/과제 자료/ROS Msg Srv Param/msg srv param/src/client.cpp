#include <ros/ros.h>
#include <assign/msg.h>
#include <assign/srv.h>
#include <cstdlib>
#include <time.h>

assign::srv srv;
ros::ServiceClient client;
ros::Publisher pubMsg;
int iter;

void calc(const ros::TimerEvent&);


int main(int argc, char **argv)
{
    iter = 0;

    ros::init(argc, argv, "Client");

    ROS_INFO("Client");

    ros::NodeHandle n;

    pubMsg = n.advertise<assign::msg>("/assign4/msg", 1);
    client = n.serviceClient<assign::srv>("calc_area");

    ros::Timer timer = n.createTimer(ros::Duration(1), calc); // 1�ʿ� �ѹ��� Service ��û

    ros::spin();

    return 0;
}

void calc(const ros::TimerEvent&){

    std::vector<int> R;      // Radius ���� ������ array R ���� 
    std::vector<float> Area; // Area ���� ������ array Area ����

    assign::msg msg;

    srand(time(NULL));

    for(int i = 0; i< 5; i++) {
        int radius = rand()%15 + 1;; // array R�� Random Radius �� push_back
        R.push_back(radius); 
    }

    srv.request.radius = R;         // Service Input

    if (client.call(srv))           // Service ��û ���� ��
    {
        ROS_INFO("Sub Area %d", iter);
        for(int i = 0; i< 5; i++) {
            ROS_INFO("%.2f", srv.response.area[i]);
            Area.push_back(srv.response.area[i]);  // array Area�� Service Response�� push_back
        }

        msg.header.stamp = ros::Time::now();
        msg.radius = R;       // msg�� radius�� array R�� ����
        msg.area = Area;      // msg�� area�� array Area�� ����

        pubMsg.publish(msg);  // msg publish


    }
    else
    {
        ROS_ERROR("Failed to call service add_two_ints");
    }

    iter += 1;

}
