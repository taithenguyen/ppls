/* 
The implementation follows the standard bag of tasks pattern.

Farmer: after the initialization, the farmer loop is in execution until all workers
have finished computing the area and the stack is empty. The loop starts with a wild-card
synchronous MPI receive function and when data is received we check whether it is a new task
or a computed area. If it is the former, we push onto the stack and if it is the latter, we
add the value to the area total. We then increment idle_count and set the corresponding 
worker slot in worker_list to 1 (meaning available). If the stack is not empty,
we iterate over the worker_list, starting from the last processed slot and look for
available workers to send new tasks. Finally, when when the stack is empty and all workers
are available, we have computed the area and the farmer process breaks the main loop and signals
all workers to exit. 

Worker: the worker loop starts with synchronous MPI receive command and waits for
input from the farmer. Once input is received, it is processed according to
the adaptive quadrature algorithm and the results are sent back to the farmer.
When an exit signal is received, the worker terminates.

MPI primitives: Synchronous blocking wild-card MPI receive is used on the farmer and helps
avoid iterating over all workers with asynchronous receive to look for results. This also
ensures synchronization with all workers. The send used requires neither the farmer nor the workers
to wait till data is received, providing it is known that it will be received eventually.

I also considered using MPI gather, but that has a drawback over the current approach - 
workers can finish their tasks at different speeds and waiting for input from all workers on 
each iteration of the farmer loop would have likely negatively impacted overall performance. 

The code is tested on a Core 2 Duo processor (2 cores) and OpenMPI 1.4.3, so usleep is not used.

$ mpirun -c 5 aquadPartA 
Area=7583461.801486

Tasks Per Process
0	1	2	3	4	
0	1679	1605	1682	1601	

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define SLEEPTIME 1

typedef struct stack_node_tag stack_node;
typedef struct stack_tag stack;

struct stack_node_tag {
	double data[2];
	stack_node *next;
};

struct stack_tag {
	stack_node *top;
};

stack *new_stack();
void free_stack(stack *);

void push(double *, stack *);
double *pop (stack *);

int is_empty(stack *);

int *tasks_per_process;

double farmer(int);

void worker(int);

int main(int argc, char **argv ) {
	int i, myid, numprocs;
	double area, a, b;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	if (numprocs < 2) {
		fprintf(stderr, "ERROR: Must have at least 2 processes to run\n");
		MPI_Finalize();
		exit(1);
	}

	if (myid == 0) { // Farmer
		// init counters
		tasks_per_process = (int *) malloc(sizeof(int)*(numprocs));
		for (i=0; i<numprocs; i++) {
			tasks_per_process[i]=0;
		}
	}

	if (myid == 0) { // Farmer
		area = farmer(numprocs);
	} 
	else { //Workers
		worker(myid);
	}

	if(myid == 0) {
		fprintf(stdout, "Area=%lf\n", area);
		fprintf(stdout, "\nTasks Per Process\n");
		for (i=0; i<numprocs; i++) {
			fprintf(stdout, "%d\t", i);
		}
		fprintf(stdout, "\n");
		for (i=0; i<numprocs; i++) {
			fprintf(stdout, "%d\t", tasks_per_process[i]);
		}
    		fprintf(stdout, "\n");
    		free(tasks_per_process);
  	}

  	MPI_Finalize();
  	return 0;
}

double farmer(int numprocs) {
	MPI_Status status;
	double buff[] = {0,0};
	int workers = numprocs - 1; // number of workers
	int idle_count = 0; // count of idle workers
	int* worker_list = (int*) malloc(sizeof(int)*(workers)); // the list of workers. 1 if available, 0 if calculating
	double result = 0;

	stack* bag;
	bag = new_stack();
	buff[0] = A;
	buff[1] = B;
	push(buff, bag);
  
	int i;
	for (i=0;i<workers;i++){
		worker_list[i] = 0;
	}
	// control loop
	do {
		MPI_Recv(buff, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // receive from workers
		idle_count++;
		worker_list[status.MPI_SOURCE - 1] = 1;
		if (status.MPI_TAG == 1) {
			result += buff[0];
		} 
		else {
			push(buff,bag);
      			MPI_Recv(buff, 2, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
      			push(buff,bag);
    		}
		int j = 0;
		while(!is_empty(bag) && idle_count>0) { // iterate over the worker_list to find idle workers and send them tasks
			if (worker_list[j]) {
				MPI_Send(pop(bag), 2, MPI_DOUBLE, j+1, 0, MPI_COMM_WORLD);
				worker_list[j] = 0;
				idle_count--;
				tasks_per_process[j+1]++;
      			}
      		j = (j+1) % workers;
    		}
	} while(!is_empty(bag) || idle_count!=workers);
	for (i=0;i<workers;i++) { // signal for exit
		buff[0] = 0;
		buff[1] = 0;
		MPI_Send(buff, 2, MPI_DOUBLE, i+1, 1, MPI_COMM_WORLD);
	}
	return result;
}

void worker(int mypid) {
	MPI_Status status;
	double buff[] = {0,0};
	MPI_Send(buff, 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	// Worker loop
	while (1) {
		MPI_Recv(buff, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG != 1) {
	    		double left = buff[0];
	    		double right = buff[1];
	    		double lrarea = (F(left) + F(right)) * (right - left) / 2;
			double mid, fmid, larea, rarea;
			mid = (left + right) / 2;
			fmid = F(mid);
			larea = (F(left) + fmid) * (mid - left) / 2;
			rarea = (fmid + F(right)) * (right - mid) / 2;
			if (fabs((larea + rarea) - lrarea) > EPSILON) {
				buff[0] = left;
				buff[1] = mid;
				MPI_Send(buff, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // task 1
				buff[0] = mid;
				buff[1] = right;
				MPI_Send(buff, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // task 2
			} else {
				buff[0] = larea + rarea;
				buff[1] = 0;
				MPI_Send(buff, 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	    		}
		}
		else {
			break; // exit 
		}
  	}
}

// creating a new stack
stack * new_stack() {
	stack *n;
	n = (stack *) malloc (sizeof(stack));
	n->top = NULL;
	return n;
}

// cleaning up after use
void free_stack(stack *s) {
	free(s);
}

// Push data to stack s, data has to be an array of 2 doubles
void push (double *data, stack *s) {
	stack_node *n;
	n = (stack_node *) malloc (sizeof(stack_node));
	n->data[0] = data[0];
	n->data[1] = data[1];
  
	if (s->top == NULL) {
		n->next = NULL;
		s->top  = n;
	} 
	else {
		n->next = s->top;
		s->top = n;
	}
}

// Pop data from stack s
double * pop (stack * s) {
	stack_node * n;
	double *data;
	if (s == NULL || s->top == NULL) {
		return NULL;
	}
	n = s->top;
	s->top = s->top->next;
	data = (double *) malloc(2*(sizeof(double)));
	data[0] = n->data[0];
	data[1] = n->data[1];
	free (n);
	return data;
}

// Check for an empty stack
int is_empty (stack * s) {
  return (s == NULL || s->top == NULL);
}
