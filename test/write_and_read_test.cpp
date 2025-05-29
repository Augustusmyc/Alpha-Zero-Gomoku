#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include<iostream>
#include<fstream>

#include <tuple>
#include <vector>


using namespace std;

int main() {
	ofstream bestand;
	vector<vector<float>> v(7, vector<float>(5));
	const char* pointer;

	bestand.open("test", ios::out | ios::binary);
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 5; j++) {
			v[i][j] = i + 0.001 * j;
		}
	}
	cout << v.size() << endl;
	size_t bytes = 35 * sizeof(float);
	for (int i = 0; i < 7; i++) {
		pointer = reinterpret_cast<const char*>(&v[i][0]);
		bestand.write(reinterpret_cast<char*>(&v[i][0]), 5 * sizeof(float));
	}
	bestand.close();


	ifstream inlezen;
	vector<vector<float>> v2(7, vector<float>(5));
	inlezen.open("test", ios::in | ios::binary);
	//char byte[8];
	//bytes = v2.size() * sizeof(v2[0]);
	for (int i = 0; i < 7; i++) {
		inlezen.read(reinterpret_cast<char*>(&v2[i][0]), 5 * sizeof(float));
	}
	
		
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 5; j++) {
			cout << v[i][j] << endl;
		}
	}
}