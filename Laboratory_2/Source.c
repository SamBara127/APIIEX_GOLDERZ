#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include <assert.h>
#include<time.h>

// лямбда функция максимума для редукции
#define max(x, y) ((x) > (y) ? (x) : (y) )

int main(int argc, char *argv[])
{
	assert(argc == 4);
	//объявляем "size of matrix", "accurancy" и "iterations_max"
	clock_t start, end;
	double result;

	start = clock();
	int size = atoi(argv[1]);
	double accur = atof(argv[2]);
	int iter_max = atoi(argv[3]);
	
	//смещение массива на 2 индекса:
    // 1 - в связи с тем что происходит обрез тензора по "правому" и "нижнему" краю(<20 а не =20)
    // 2 - в свзи с тем что при пробеге в циклах по тензору по "правому" и "нижнему" краю будет браться
    // мнимый ноль на границах чтобы не выйти за предел тензора (см схему ниже под кодом)
	int bias_1 = 2;
	// создание двумерного массива - тензора с учетом добавления обреза
	// создание двух матриц нужно для того чтобы можно было вычитать ячейки проработанного массива
	// и находить ошибку и быстро переобновлять матрицу
	double **main_arr = (double**)malloc((size + bias_1)* sizeof(double*));
	double **main_arr2 = (double**)malloc((size + bias_1)* sizeof(double*));
	
	for (int i=0;i<size + bias_1;i++)
	{
		main_arr[i] = (double*)malloc((size + bias_1) * sizeof(double));
		main_arr2[i] = (double*)malloc((size + bias_1) * sizeof(double));
	}
	// объявление переменной отсчета итерации и градиента = шага нарастания от одной вершины до другой  
	int iter = 0;
	double gradient = 10.0 / size;
	// объявление переменной ошибки 
	double error = 1.0;
    // двойной указатель на массив как контейнер обмена данными между 2 тензорами
    double **ptr;

	/* Так как у нас НЕструктурированая модель данных мы выделяем память на ГПУ под размеры соответствующие
		нашим массивам (только выделяем с помощью create) и ТОЛЬКО копируем данные (copyin) на ГПУ ускоритель и заполняем этот массив на ГПУ памяти
	*/
	#pragma acc enter data create(main_arr[0:size+bias_1][0:size+bias_1],main_arr2[0:size+bias_1][0:size+bias_1]) copyin(size, gradient, bias_1)
    {
        /*  заполнение ребер тензоров по градиенту, возьмем parallel и independent для небольшого ускорения
            у нас независимые итерации поэтому можно все растянуть на блоко-ниточную архитектуру сетки и вдобавок
            запустить асинхронно ядро (пусть будет номер 2, берем любое незанятое - async(2)) на заполнение первого массива
            чтобы эти два тензора заполнялись на разных ядрах(грубо говоря сеток) параллельно и потом синхронизируем
            их чтобы они поступали на обработку уже заполненые оба, такой подход ускорит работу в данной области
        */
        #pragma acc parallel async(2) // BOOST
        {
            #pragma acc loop independent
            for (int i=0;i<size + bias_1;i++)
            {
                main_arr[i][0] = 10 + gradient*i;
                main_arr[0][i] = 10 + gradient*i;
                main_arr[size][i] = 20 + gradient*i;
                main_arr[i][size] = 20 + gradient*i;
            }
        }
        #pragma acc parallel
        {
            #pragma acc loop independent
            for (int i=0;i<size + bias_1;i++)
            {
                main_arr2[i][0] = 10 + gradient*i;
                main_arr2[0][i] = 10 + gradient*i;
                main_arr2[size][i] = 20 + gradient*i;
                main_arr2[i][size] = 20 + gradient*i;
            }
        }
    
        /*
            На данном этапе перенесенные переменные выше в диерктиве copyin  автоматически
            удаляются из видеопамяти за ненадобностью в дальнейших вычислениях
            поэтому нужно проинизиализировать повторно но уже только переменную error.
            Так как у нас сама цель просто ее выделить в памяти для дальнейшей обработки
            то можно грубо говоря писать любую директиву выделения(копирования):
            copyin; copyout; create..., 
            все равно значение ее мы будем по итогу экспортировать на ЦПУ RAM в директиве:
            #pragma acc update host(error),
            но мы возьмем copyout чтобы в конце на хосте вывести конечное значение error для удобства
        */ 

        // синхронизируем верхнее второе ядро заполнения и переходим далее
        #pragma acc wait(2)

        #pragma acc data copyout(error)

        /*
            начало цикла внутреннего формирования тензора, где прерыванием будет предел 
            итераций т.е.грубо говоря колличество "Эпох"; или предел разницы между двумя ячейками
            тензоров main_arr и main_arr2
        */ 
        while ((error > accur) && (iter<iter_max))
        {
            iter++;
            /*
                мы разбили обновление значения ошибки через интервал в 150 итераций, это было сделано для того
                чтобы мы могли выполнять передачу error на хост без задержки во времени, в противном случае 
                так как процедура #pragma acc update host(error) работает с передачей в хост-память значение 
                error, то такая процедура:
                1) - выполняется последовательно на ядре
                2) - из за первого пункта сильно тормозит работу ядра на основных вычислениях нашего тензора
                Поэтому число итерации 150 было выбрано как оптимальное, исходя из показаний NSightSystems
                и вывода на консоли, ведь если мы вызовем эту директиву отправки в память хоста и недождавшись 
                завершения этой отправки посылаем следующий запрос на отправку, то программа грубо говоря
                забьется этими очередями и начнет сильно тормозить, да и само увеличение интервала
                увеличивает скорость работы хотя бы потому что мы постоянно не вызваем эту директиву 
                а только через промежутки как у нас в 150 итераций :)
            */
            if ((iter % 150 == 0) || (iter == 1))
            {
                /*
                    обнуляем ошибку и создаем для каждой нитки потоковой ее дополнительно, чтобы при 
                    распараллеливании они все имели нулевое значение
                    также мы параллелим все под асинхронный режим на первое свободное ядро,
                    опять же все из за тормозов работы с передачей памяти в #pragma acc update host(error)
                    такая работа будет независеть от работы передачи в память хоста и выполняться параллельно с ней 
                */
                #pragma acc parallel async(1) // kernels -> parallel async(1) 
                error = 0.0;

                // инициализируем уже существующие в видеопамяти массивы main_arr и main_arr2
                #pragma acc data present(main_arr, main_arr2)
                /*
                    На данном этапе мы распалаллеливаем ниже цикл, причем цикл у нас объединен
                    из 2 в 1 с помощью gang vector(кол-во блоков) - где указывают что цикл 
                    имеет 2 вектора вычисления которые могут обоюдно вычилсятся параллельно
                    по сути если смотреть аппаратно мы перенесли наши 2 цикла на двумерный тензор ГПУ,
                    в котором абсцисса - блоки или итерации первого цикла, а ордината это нитки или 
                    итерации второго цикла. Все они на этом тензоре одновременно проходят по своим потокам
                    и таким образом мы грубо говоря за 1 проход пробежали все итерации цикла на тензоре main_arr2
                    а это очень сильно ускоряет работу. см схему ниже
                */
                #pragma acc parallel async(1) // kernels -> parallel async(1) 
                {
                    /*
                        reduction - нужна для ускорения нахождения в нашем случае максимума ошибки
                        архитектура работы - создает массив(вектор) где каждый элемент в ней это вычисленное
                        значение максимума ошибки на потоке, номер которого совпадает с индексом этого массива,
                        что то похожее на хеш-таблицу, когда все потоки найдут в своей итерации значение ошибки и
                        массив заполнится и сократится, получив локальные максимумы в каждом блоке и объединив их в 1 вектор,
                        далее компиллятор сгенерирует новое ядро где финальной работой будет вычисление максимума
                        из полученного вектора(метод похож на параллельное суммирование)
                    */ 
                    #pragma acc loop gang vector() reduction(max:error)
                    for (int j = 1; j < size + 1; j++)
                    {
                        #pragma acc loop gang vector() 
                        for (int i = 1; i < size + 1; i++)
                        {
                            main_arr2[i][j] = 0.25 * (main_arr[i+1][j] + main_arr[i-1][j] + main_arr[i][j-1] + main_arr[i][j+1]);
                            error = max(error, main_arr2[i][j] - main_arr[i][j]);
                        }
                    }
                }
            }
            else 
            {
                #pragma acc data present(main_arr, main_arr2)
                #pragma acc parallel async(1) // kernels -> parallel async(1) 
                {
                    #pragma acc loop gang vector()
                    for (int j = 1; j < size + 1; j++)
                    {
                        #pragma acc loop gang vector()
                        for (int i = 1; i < size + 1; i++)
                        {
                            main_arr2[i][j] = 0.25 * (main_arr[i+1][j] + main_arr[i-1][j] + main_arr[i][j-1] + main_arr[i][j+1]);
                        }
                    }
                }
            }
            // меняем указатели массивов чтоб обновить полученные значения на первый массив
            ptr = main_arr;
            main_arr = main_arr2;
            main_arr2 = ptr;

            if ((iter % 150 == 0) || (iter == 1))
            {
                //синхронизируем ядро для подготовки error к отправке, иначе error будет меняться 
                //на показаниях нелинейно по отношению к итерации
                #pragma acc wait(1)
                // отправка на хост нашей error, одно из тормозящих мест кода.
                #pragma acc update host(error)
                printf("Iteration - %d ; Error = %0.14lf\n", iter, error);
            }

        }
    }

	end = clock();
    result  = (double)(end-start);
    result = result / 1000000;
    printf("Iteration - %d ; Error = %0.14lf\n", iter, error);
    printf("Time = %f Seconds\n", result);

	return 0;

}

/*  схема строения тензора: 8 x 8

g(i) = 10 + gradient*i
f(i) = 20 + gradient*i
i = index

i = {0 .. 7} - 8 элементов
i = i + 2 = {0 .. 9} элементов - 7-ой элемент не равен значению вершины тензора
поэтому мы делаем +1 смещение для получения полного тензора без обреза по краям снизу и справа (index = 8);
далее делаем еще одно смещение +1 где мы получаем два мнимых края тензора c рандомными значеним 
чтобы при применении уравнения теплопроводности на точках (x, 8) и (8, x) где x -> {0 .. 8}
мы не заходили за пределы тензора или за область допустимой памяти:

main_arr2[i][j] = 0.25 * (main_arr[i+1][j] + main_arr[i-1][j] + main_arr[i][j-1] + main_arr[i][j+1]);


_i_|_0 _|_1 _|_2 _|_3 _|_4 _|_5 _|_6 _|_7 _|_8 _|_9 _
_0_|_10_|_1g_|_2g_|_3g_|_4g_|_5g_|_6g_|_7g_|_20_|_??_
_1_|_1g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_1f_|_??_
_2_|_2g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_2f_|_??_
_3_|_3g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_3f_|_??_
_4_|_4g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_4f_|_??_
_5_|_5g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_5f_|_??_
_6_|_6g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_6f_|_??_
_7_|_7g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_7f_|_??_
_8_|_20_|_1f_|_2f_|_3f_|_4f_|_5f_|_6f_|_7f_|_30_|_??_
_9_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_

*/
