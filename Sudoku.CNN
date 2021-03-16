using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Keras.Layers;
using Keras.Utils;
using Numpy;
using Numpy.Models;
using Python.Runtime;
using static Python.Runtime.Py;
using static CNN.BaseLayer.Dense;
using static CNN.BaseModel;

namespace CNN
{

    public class Keras : IDisposable
    {
        public static Keras Instance => _instance.Value;

        public static dynamic sys;

        public static bool DisablePySysConsoleLog { get; set; } = false;

        public static object Setup { get; private set; }

        public static bool alreadyDisabled = false;

        public static object InstallAndImport(object kerasModule)
        {
            throw new NotImplementedException();
        }

        void IDisposable.Dispose()
        {
            throw new NotImplementedException();
        }

        private static Lazy<Keras> _instance = new Lazy<Keras>(() =>
        {
            var instance = new Keras();
            instance.keras = InstallAndImport(Setup.kerasModule);
        }
        );

        
    }

    public class BaseModel
    {
        public object PyInstance { get; private set; }
        public object Instance { get; private set; }

        internal class PyObject
        {
        }

        public class Sequential : BaseModel
        {

            internal Sequential(PyObject obj)
            {
            PyInstance = obj;
            }

            public Sequential()
            {
            PyInstance = Instance.keras.models.Sequential();
            }

            public Sequential(BaseLayer[] layers) : this()
            {
                foreach (var item in layers)
                {
                Add(item.PyInstance);
                }
            }

            public void Add(BaseLayer layer)
            {
                PyInstance.add(layer: layer.PyInstance);
            }

        }
    }

    public class BaseLayer
    {
        public BaseLayer PyInstance { get; internal set; }
        public object Instance { get; private set; }

        public class Shape
        {
            public int item1;
            public int item2;

            public Shape(int item1, int item2)
            {
                this.item1 = item1;
                this.item2 = item2;
            }
        }

        public void Init()
        {
            throw new NotImplementedException();
        }



        public class Conv2D : BaseLayer
        {

            public Conv2D(int filters, Tuple<int, int> kernel_size, string padding = "valid", string activation = "", Shape input_shape = null)
            {
                Parameters["filters"] = filters;
                Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
                Parameters["padding"] = padding;
                Parameters["activation"] = activation;
                Parameters["input_shape"] = input_shape;

                PyInstance = Instance.keras.layers.Conv2D;
                Init();
            }
            
        }

        public class BatchNormalization : BaseLayer
        {
            public BatchNormalization()
            {
                PyInstance = Instance.keras.layers.BatchNormalization();
            }

        }

        public class Flatten : BaseLayer
        {
            public Flatten()
            {
                PyInstance = Instance.keras.layers.Flatten;
                Init();
            }

            
        }

        public class Dense : BaseLayer
        {
            public Dense(int units, string activation = "", Shape input_shape = null)
            {
                this["units"] = units;
                this["activation"] = activation;
                Parameters["input_shape"] = input_shape;
                PyInstance = Instance.keras.layers.Dense;
                Init();
            }


        public class Reshape : BaseLayer
        {
            public Reshape(Shape target_shape, Shape input_shape = null)
            {
                Parameters["target_shape"] = target_shape;
                Parameters["input_shape"] = input_shape;
                PyInstance = Instance.keras.layers.Reshape;
                Init();
            }

        }

        public class Activation : BaseLayer
        {
            public Activation(string act, Shape input_shape = null)
            {
                 Parameters["activation"] = act;
                 Parameters["input_shape"] = input_shape;
                 PyInstance = Instance.keras.layers.Activation;
                 Init();
            }
        }

    }



    public static void Run()
    {
        // Parametres d'entrainement
        int batch_size = 128;
        int epochs = 12;

        // Chargement des donnees et mise en forme


        // Modele CNN
        var model = new Sequential();

        model.Add(new Conv2D(64, kernel_size: (3, 3).ToTuple(), padding: 'same', activation: "relu", input_shape: (9, 9, 1));
        model.Add(new BatchNormalization());
        model.Add(new Conv2D(64, (3, 3).ToTuple(), padding: 'same', activation: "relu"));
        model.Add(new BatchNormalization());
        model.Add(new Conv2D(128, (1, 1).ToTuple(), padding: 'same', activation: "relu"));


        model.Add(new Flatten());
        model.Add(new Dense(81*9));
        model.Add(new Reshape((-1, 9)));
        model.Add(new Activation('softmax'));

        // Compile le modele
        model.Compile(loss: "categorical_crossentropy",
          optimizer: new Adadelta(), metrics: new string[] { "accuracy" });

        // Entrainement du modele
        model.Fit(x_train, y_train,
                  batch_size: batch_size,
                  epochs: epochs,
                  verbose: 1,
                  validation_data: new NDarray[] { x_test, y_test });


        // Accuracy du modele
        var score = model.Evaluate(x_test, y_test, verbose: 0);
        Console.WriteLine("Test loss:" + score[0]);
        Console.WriteLine("Test accuracy:" + score[1]);


    }
}
