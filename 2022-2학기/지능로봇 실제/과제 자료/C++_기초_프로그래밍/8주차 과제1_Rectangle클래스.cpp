#include <iostream>
using namespace std;


// Ŭ���� ����
class Rectangle {
public:
	int width;
	int height;
	int getArea();
};

// Ŭ���� ����
int Rectangle::getArea() {
	return width * height; // �簢���� ���� return
}

int main() {

	Rectangle rect;
	rect.width = 3;
	rect.height = 5;
	cout << "�簢���� ������ " << rect.getArea() << endl;

}