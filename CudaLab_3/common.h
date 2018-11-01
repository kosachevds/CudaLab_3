#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <iostream>

template <typename T>
void WriteVector(std::vector<T> const& values, std::ostream& out)
{
    for (auto const& item: values) {
        out << item << " ";
    }
    out << std::endl;
}


#endif // COMMON_H
