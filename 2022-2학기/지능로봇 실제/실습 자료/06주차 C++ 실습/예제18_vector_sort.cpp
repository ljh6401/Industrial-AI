#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> v;

	std::cout << "���� 5���� �Է��ϼ��� -> ";

	for (int i = 0; i < 5; i++) {
		int n;
		std::cin >> n;
		v.push_back(n);
	}

	std::sort(v.begin(), v.end());  // sort �Լ��� �̿��ؼ� v.begin() ~ v.end() ������ ���� ������������ ����

	std::vector<int>::iterator it;  // ���� ���� ���Ҹ� Ž���ϴ� iterator ����

	for (it = v.begin(); it != v.end(); it++) {
		std::cout << *it << ' ';    // ������ ��� ���� ���
	}
	std::cout << std::endl;
}