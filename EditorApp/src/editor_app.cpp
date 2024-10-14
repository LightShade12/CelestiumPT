#include "application.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
	fprintf(stderr, "Running from: %s\n", argv[0]);
	{
		Application app;
		app.run();
	}
}