#include <iostream>
using namespace std;

// Ŭ���� ����
class Circle {
public:
	int radius;
	double getArea();
};

// Ŭ���� ����
double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;                   // donut ��ü ����
	donut.radius = 1;               // donut ��� ���� ����
	double area = donut.getArea();  // donut ��� �Լ� ȣ��
	cout << "donut ������ " << area << endl;

	Circle pizza;
	pizza.radius = 30;
	area = pizza.getArea();
	cout << "pizza ������ " << area << endl;

}
