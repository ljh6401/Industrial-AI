#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> v;

	v.push_back(1);
	v.push_back(2);
	v.push_back(3);

	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;

	v[0] = 10;    // �ش� ���� �� ����
	int n = v[2];
	v.at(2) = 5;  // �ش� ���� �� ����

	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}