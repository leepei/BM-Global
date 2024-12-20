#include "util.h"
#include "pmf.h"
#include <cstring>
//extern "C" {
//	#include "declarations.h"
//}

bool with_weights;

void exit_with_help()
{
	printf(
	"Usage: omp-pmf-train [options] data_dir [model_filename]\n"
	"options:\n"
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -E epsilon2 : set outer termination criterion epsilon (default 1e-6)\n"     
	"    -p do_predict: do prediction or not (default 0)\n"    
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n"
	"    -i file_name: use file name as initialization\n"    
	);
	exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	parameter param;   // default values have been set by the constructor 
	with_weights = false;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'E':
				param.outereps = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			case 'i':
				param.init_file = strdup(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0) 
		param.verbose = 1;

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') 
			*p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	return param;
}


void run_algo(parameter &param, const char* input_file_name, const char* model_file_name=NULL){
	smat_t R;
	smat_t R2;
	mat_t W,H;
	testset_t T, T2;

	FILE *model_fp = NULL;

	if(model_file_name) {
		model_fp = fopen(model_file_name, "wb");
		if(model_fp == NULL)
		{
			fprintf(stderr,"can't open output file %s\n",model_file_name);
			exit(1);
		}
	}

	load(input_file_name,R,T, with_weights);
	load(input_file_name,R2,T, with_weights);
	// W, H  here are k*m, k*n
	initial_col(W, param.k, R.rows);
	initial_col(H, param.k, R.cols);

	puts("starts!");
	double time = omp_get_wtime();
	polyMF_SS(R, W, H, T, param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	if(model_fp) {
		save_mat_t(W,model_fp,false);
		save_mat_t(H,model_fp,false);
		fclose(model_fp);
	}
	return ;
}


int main(int argc, char* argv[]){
	char input_file_name[1024];
	char model_file_name[1024];
	parameter param = parse_command_line(argc, argv, input_file_name, model_file_name);

	run_algo(param, input_file_name, model_file_name);
	//switch (param.solver_type){
	//	case POLY:
	//		run_poly(param, input_file_name, model_file_name);
	//		break;
	//	default:
	//		fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
	//		break;
	//}
	return 0;
}

