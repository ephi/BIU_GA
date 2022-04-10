// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <cstdint>
#include "PythonGALib.h"
#define BOOST_PYTHON_STATIC_LIB
#include "boost/python.hpp" 
using namespace boost::python;
using namespace std;

#define MAX_POPULATION_SIZE 300
namespace PythonGALib
{
	// Avoid cluttering the global namespace. 

     uint64_t to_gray(uint64_t n) {
		return (n ^ (n >> 1));
     }
    // Return the inverse Gray code of n
    uint64_t inverse_gray(uint64_t n)
    {
        // This is the more complicated direction: In hierarchical 
        // stages, starting with a one-bit right shift, cause each 
        // bit to be XORed with all more significant bits.
        int ish = 1;
        uint64_t ans = n;
        uint64_t idiv = 0;

        for (;;) {
            idiv = (ans >> ish);
            ans ^= idiv;
            if (idiv <= 1 || ish == 16)
                return ans;
            ish <<= 1;
        }
        return ans;
    }

}

BOOST_PYTHON_MODULE(PythonGALib)
{
	using namespace PythonGALib;
	// Add regular functions to the module. 
	boost::python::def("to_gray", to_gray);
	boost::python::def("inverse_gray", inverse_gray);
}
