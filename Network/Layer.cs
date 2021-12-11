using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab3
{
    public class Layer
    {
        //Размер слоя
        public int Size;
        //Нейроны
        public Neuron[] Neurons;
        //Веса
        public double[][] Weights;

        public Layer() { }

        public void Init(int size, int next) 
        {
            Size = size;
            Neurons = new Neuron[size];
            Weights = new double[size][];

            for ( int i = 0; i < size; i++) 
            {
                Neurons[i] = new Neuron();
                Weights[i] = new double[next];
            }
        }
    }
}
