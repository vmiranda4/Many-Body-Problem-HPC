#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>


#define G 6.67430e-11 //constante gravitacional
#define MAX_PARTICLE_MASS 2000.0 //masa maxima
#define MAX_POSITION 100.0 //posicion maxima
#define MAX_VELOCITY 5.0 //velocidad maxima
#define SOFTENING_FACTOR 1.0 //factor para evitar magnitud de la fuerza gravitacional muy alta

//estructura de la particula, para facilitar el acceso a sus propiedades
typedef struct {
    double x, y; //posicion
    double vx, vy; //velocidad
    double fx, fy; //fuerza
    double masa; //masa
} Particula;


//para calcular una de las componentes del vector que une las particulas
double calcular_r_i(double x1, double x2){
    return x2 - x1;
}


//calcula la magnitud del vector r(la distancia entre dos particulas)
double calcular_distancia_entre_particulas(double x1, double x2, double y1, double y2){
    return sqrt(calcular_r_i(x1, x2) * calcular_r_i(x1, x2) + calcular_r_i(y1, y2) * calcular_r_i(y1, y2));
}


//calcular la parte escalar de la fuerza gravitacional (m1*m2)/|r|*|r|
double calcular_fuerza_escalar(double masa1, double masa2, double distancia){
    return G * masa1 * masa2 / (distancia * distancia);
}

//calcula la fuerza de todas las particulas del conjunto
void calcular_fuerza_total(Particula *particulas, int particula_id, int n){
    particulas[particula_id].fx = 0.0;
    particulas[particula_id].fy = 0.0;

    for(int i = 0; i < n; i++){
        if(particula_id != i){
            double dx = calcular_r_i(particulas[particula_id].x, particulas[i].x);
            double dy = calcular_r_i(particulas[particula_id].y, particulas[i].y);
            double distancia = calcular_distancia_entre_particulas(particulas[particula_id].x, particulas[i].x, particulas[particula_id].y, particulas[i].y);

            // Para evitar distancias muy pequeñas que afecten la fuerza gravitacional
            distancia = fmax(distancia, SOFTENING_FACTOR);

            double fuerza = calcular_fuerza_escalar(particulas[particula_id].masa, particulas[i].masa, distancia);
            particulas[particula_id].fx += fuerza * (dx / distancia);
            particulas[particula_id].fy += fuerza * (dy / distancia);
        }
    }
}


//de acuerdo con la fuerza calculada para cada particula se actualiza su velocidad y posicion
void actualizar_velocidad_y_posicion(Particula *particulas, int n, double dt){
    for(int i = 0; i < n; i++){
        particulas[i].vx += (particulas[i].fx / particulas[i].masa) * dt;
        particulas[i].vy += (particulas[i].fy / particulas[i].masa) * dt;

        //particulas[i].x += particulas[i].vx * dt;
        //particulas[i].y += particulas[i].vy * dt;
        double newX = particulas[i].x + particulas[i].vx * dt;
        double newY = particulas[i].y + particulas[i].vy * dt;
        
        //se verifica que las coordenadas se mantengan dentro del rango estipulado
        if(newX< -MAX_POSITION){
            particulas[i].x= -MAX_POSITION+0.1;
        }
        else if(newX>MAX_POSITION){
            particulas[i].x= MAX_POSITION-0.1;
        }
        else{
            particulas[i].x=newX;
        }
        if(newY< -MAX_POSITION){
            particulas[i].y= -MAX_POSITION+0.1;
        }
        else if(newY>MAX_POSITION){
            particulas[i].y= MAX_POSITION-0.1;
        }
        else{
            particulas[i].y=newY;
        }
//*/        
    }
}


// Obtener limite inferior de bloques de datos a repartir por proceso
int get_lower_limit(int rank, int size, int n) {
    return rank*n/size;
}

// Obtener limite superior de bloques de datos a repartir por proceso
int get_upper_limit(int rank, int size, int n){
    return get_lower_limit(rank + 1, size, n);
}

// Obtener cantidad de datos que le corresponde a cada proceso
int get_size_per_proc(int rank, int size, int n){
    return get_upper_limit(rank, size, n) - get_lower_limit(rank, size, n);
}


//inicializa las particulas con valores entre los rangos dados
Particula *inicializar_particulas(int n){
    //srand(time(NULL));
    Particula *particulas = (Particula *)malloc(n * sizeof(Particula));
    for(int i = 0; i < n; i++){
        particulas[i].x = (((double)rand() / RAND_MAX) * 2 - 1) * MAX_POSITION; 
        particulas[i].y = (((double)rand() / RAND_MAX) * 2 - 1) * MAX_POSITION; 
        particulas[i].vx = (((double)rand() / RAND_MAX) * 2 - 1) * MAX_VELOCITY; 
        particulas[i].vy = (((double)rand() / RAND_MAX) * 2 - 1) * MAX_VELOCITY; 
        particulas[i].masa = ((double)rand() / RAND_MAX) * MAX_PARTICLE_MASS;
    }
    return particulas;
}

//es necesario crear un MPI datatype para enviar las particulas entre procesos
MPI_Datatype create_particula_datatype() {
    const int nitems = 7;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[7];

    offsets[0] = offsetof(Particula, x);
    offsets[1] = offsetof(Particula, y);
    offsets[2] = offsetof(Particula, vx);
    offsets[3] = offsetof(Particula, vy);
    offsets[4] = offsetof(Particula, fx);
    offsets[5] = offsetof(Particula, fy);
    offsets[6] = offsetof(Particula, masa);

    MPI_Datatype MPI_PARTICULA;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_PARTICULA);
    MPI_Type_commit(&MPI_PARTICULA);

    return MPI_PARTICULA;
}


int main(int argc, char ** argv){

    int rank, size;
    int n=500; //numero de particulas total
    double dt=0.01; //paso de tiempo
    int step = 10000; //cantidad de pasos
    int local_n; //numero de particulas por proceso
    int local_min; //indice menor que le corresponde a cada proceso del arreglo de particulas totales
    int local_max; //indice mayor que le corresponde a cada proceso del arreglo de particulas totales
    Particula *vector_particulas; //vector que contiene todas las particulas
    Particula *vector_local_particulas; //vector que contiene las particulas de cada proceso

    //FILE * particulas_file; // si se quisieran leer las particulas desde un archivo
    FILE * posiciones_file;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Datatype MPI_PARTICULA = create_particula_datatype();

    int sendcounts[size];
    int displs[size];

    srand(rank + time(NULL)); //se necesita una semilla diferente para cada proceso porque si no, se inicializan las particulas de todos los procesos con los mismos valores

    //se asigna memoria dinamica al vector que tiene todas las particulas y es donde cada proceso recibe el Allgatherv
    vector_particulas = (Particula *)malloc(n * sizeof(Particula));


    //el ciclo calcula los arreglos que indican el numero de particulas que va a comunicar cada procesador y los desplazamientos en el vector donde se guardan los datos
    for(int i=0; i<size; i++){
        sendcounts[i]= get_size_per_proc(i, size, n);
        displs[i]=get_lower_limit(i,size,n);
    }

    local_n = get_size_per_proc(rank, size, n); //numero de particulas correspondientes a cada proceso
    local_min= get_lower_limit(rank, size, n); //indice menor de particulas que cada proceso tiene en el vector general
    local_max = get_upper_limit(rank, size, n);//indice mayor de particulas que cada proceso tiene en el vector general


    vector_local_particulas=inicializar_particulas(local_n); //se inicializan las particulas y se guardan en los vectores de cada proceso

    //luego de que cada proceso tiene su vector local, se hace un Allgatherv para reunir todos estos vectores y que a su vez todos los procesos tengan el total de particulas para asi poder calcular las fuerzas de las particulas que le corresponden
    MPI_Allgatherv(vector_local_particulas,local_n, MPI_PARTICULA, vector_particulas,sendcounts,displs,MPI_PARTICULA, MPI_COMM_WORLD);

    //ya que todos los procesos tienen el vector completo de particulas, solo el proceso 0 se encarga de guardar las posiciones en un archivo
    if(rank==0){
        posiciones_file=fopen("posiciones_particulas.txt", "w");
        for(int i=0; i<n; i++){
            fprintf(posiciones_file,"%.2f %.2f\n",vector_particulas[i].x, vector_particulas[i].y);
        }
        fprintf(posiciones_file, "\n");
        //fclose(posiciones_file);
    }


    //se realiza la actualizacion de todos los datos, esto se hace el numero de pasos que se asignó anteriormente
    for(int i=0; i<step; i++){
        for(int j=local_min; j<local_max;j++){
            calcular_fuerza_total(vector_particulas, j,n); //primero se calculan las fuerzas
        }
        for (int k=0; k<local_n; k++){
            vector_local_particulas[k]=vector_particulas[local_min+k]; //separa los valores calculados en el arreglo local
        }
        actualizar_velocidad_y_posicion(vector_local_particulas, local_n,dt); 

        //se comparten los valores actualizados
        MPI_Allgatherv(vector_local_particulas,local_n, MPI_PARTICULA, vector_particulas,sendcounts,displs,MPI_PARTICULA, MPI_COMM_WORLD);

        //el proceso 0 guarda las posiciones en el archivo
        if(rank==0){
            for(int i=0; i<n; i++){
                fprintf(posiciones_file,"%.2f %.2f\n",vector_particulas[i].x, vector_particulas[i].y);
            }
            fprintf(posiciones_file, "\n");
        }
    }

    if(rank==0){
        fclose(posiciones_file);
    }

    free(vector_particulas);
    free(vector_local_particulas);
    MPI_Type_free(&MPI_PARTICULA); // Free the MPI datatype
    MPI_Finalize();
    return 0;

}

