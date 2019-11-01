/*
Demonstrator for
1) instantiating cpp objects from python
2) retrieving properties from the cpp object

Compile Using:
1) clib % g++ -c -fPIC demo_class.cpp -o lib_demo_class.o
2) clib % g++ -shared -Wl.-soname,lib_demo_class.so -o lib_demo_class.so lib_demo_class.o

Call
clib % python demo_class.py

*/

#include <iostream>

class MyClass {
    public:

        int p;
        int q;
        float* array;

        // default constructor
        MyClass() {
            std::cout << "MyClass constructor - default" << std::endl;
            p = -1;
            q = -1;
        }

        // parametrized constructor
        MyClass(int p_in, int q_in) {
            std::cout << "MyClass constructor - p: " << p_in << " and q: " << q_in << std::endl;
            p = p_in;
            q = q_in;
        }

        // destructor
        ~MyClass() {
            std::cout << "MyClass destructor" << std::endl;
        }

        int get_p() { return p; }
        int get_q() { return q; }
        int calc() {
            if ((p == -1) && (q == -1)){ return -2; }
            else if ((p == -1) || (q == -1)) { return -1; }
            else { return p * q + p % q; }
        }

        void set_array(float a, float b, float c) {
            array = new float[3];
            array[0] = a;
            array[1] = b;
            array[2] = c;
        }
        float* get_array() { return array; }


};

extern "C" {
    MyClass* NewMyClass(int p, int q){ return new MyClass(p, q); }
    MyClass* DefaultMyClass(){ return new MyClass(); }
    int GetP(MyClass* myClass){ return myClass->get_p(); }
    int GetQ(MyClass* myClass){ return myClass->get_q(); }
    int Calc(MyClass* myClass){ return myClass->calc(); }
    void Delete(MyClass* myClass){ delete myClass; }
    void SetArray(MyClass* myClass, float a, float b, float c){ myClass->set_array(a, b, c); }
    float* GetArray(MyClass* myClass){ return myClass->get_array(); }
}
