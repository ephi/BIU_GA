#pragma once
namespace PythonGALib {
	__declspec(dllexport) uint64_t inverse_gray(uint64_t n);
	//Takes in a group of chromosomes (each is 32bit) and selects 2 by fitness_func f
	__declspec(dllexport) uint64_t to_gray(uint64_t n);
}