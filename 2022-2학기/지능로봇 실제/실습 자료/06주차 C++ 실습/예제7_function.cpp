#include <iostream>
using namespace std;

int adder(int a, int b) {
	int sum;
	sum = a + b;
	return sum;
}

int main() {

	int n = adder(24567, 98374); // adder �Լ� ȣ��
	cout << "24567�� 98374�� ���� " << n << "\n";

	int a, b;
	cout << "�ΰ��� ������ �Է��ϼ��� -> ";
	cin >> a >> b;

	n = adder(a, b);
	cout << a << " ��" << b << " �� ���� " << n << " �Դϴ�.";

}

