#include <cstdlib>
#include <ctime>

extern void task1();

int main()
{
    srand(time(nullptr));
    task1();
    return 0;
}
