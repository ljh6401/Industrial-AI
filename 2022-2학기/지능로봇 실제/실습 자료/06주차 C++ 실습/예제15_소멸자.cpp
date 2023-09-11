#include <iostream>
using namespace std;

class Circle {
public:
	int radius;
	Circle();      // �Ű������� ���� ������
	Circle(int r); // �Ű������� �ִ� ������
	~Circle();     // �Ҹ��� ��ü�� �Ҹ�Ǵ� �������� �ڵ����� ȣ�� / �ϳ��� ����
	double getArea();
};

Circle::Circle() {
	radius = 1;
	cout << "������ : " << radius << ", �� ����" << endl;
}

Circle::Circle(int r) {
	radius = r;
	cout << "������ : " << radius << ", �� ����" << endl;
}

Circle:: ~Circle() {
	cout << "������ : " << radius << ", �� �Ҹ�" << endl;
}

double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;     // �Ű������� ���� ������ ȣ��
	Circle pizza(30); // �Ű������� �ִ� ������ ȣ��

	return 0;         // �����Լ� ���� �� donut, pizza ��ü �Ҹ� / ������ �ݴ������ �Ҹ�

}