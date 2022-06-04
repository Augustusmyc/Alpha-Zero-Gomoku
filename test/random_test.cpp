#include <iostream>
#include <random>

using namespace std;

int main() {
    std::vector<int> rand_order;
    for (int i = 0; i < 10; i++) {
        rand_order.push_back(i);
    }

    std::vector<float> rand_order2;
    for (int i = 0; i < 10; i++) {
        rand_order2.push_back((float)i/100);
    }
    unsigned seed = 0;
    auto e = default_random_engine(seed);
    shuffle(rand_order.begin(), rand_order.end(), e);
    e = default_random_engine(seed);
    shuffle(rand_order2.begin(), rand_order2.end(), e);
    for (int i = 0; i < rand_order.size(); i++) {
        cout << rand_order[i] << endl;
    }
    for (int i = 0; i < rand_order2.size(); i++) {
        cout << rand_order2[i] << endl;
    }
}
