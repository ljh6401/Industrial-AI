#include <iostream>
using namespace std;

int main() {
	int n[10];
	int i;
	int* p;

	for (i = 0; i < 10; i++) {
		*(n + i) = i * 3;         // �迭 �̸� = �ּ�
	}

	p = n;                        // ������ p �� �迭�� �ּҰ� �ο�
	for (i = 0; i < 10; i++) {
		cout << *(p + i) << ' ';  // p�� �迭�� ���ҿ� ����
	}
	cout << '\n';

	for (i = 0; i < 10; i++) {
		*p = *p + 2;              // �迭�� ���Ұ��� 2�� �����ֱ�
		p++;                      // ���� �ּҷ� �̵�
	}

	for (i = 0; i < 10; i++) {
		cout << n[i] << ' ';
	}

}