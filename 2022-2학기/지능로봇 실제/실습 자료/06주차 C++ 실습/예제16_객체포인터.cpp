#include <iostream>
using namespace std;

class Circle {
	int radius;
public:
	Circle() { radius = 1; }        // Ŭ���� ���� �� ����?
	Circle(int r) { radius = r; }
	double getArea();
};

double Circle::getArea() {
	return 3.14 * radius * radius;
}

int main() {

	Circle donut;
	Circle pizza(30);

	// ��ü �̸����� ��� ����
	cout << donut.getArea() << endl;

	// ��ü �����ͷ� ��� ����
	Circle* p;
	p = &donut;
	cout << p->getArea() << endl;
	cout << (*p).getArea() << endl;

	p = &pizza;
	cout << p->getArea() << endl;
	cout << (*p).getArea() << endl;

}