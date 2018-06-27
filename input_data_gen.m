clear all
clc

video_samples_file = 'data/targets/test.hdf';
output_file_name = 'data/input/test.hdf';

%% Load bouncing balls data
hinfo = hdf5info(video_samples_file);
Data = hdf5read(hinfo.GroupHierarchy.Datasets(1));
Data=permute(Data,[4, 2, 3, 1]);

%% Set N: Frames dim1, Nx: Frames dim2, Nt: #frames, Np: ... 
[N, D_x,D_y,D_t]=size(Data);

C=randn(D_x, D_y);         %%%%%%%%%%%%%% rand mask pattern with Gaussian distribution

%% Loop over video samples to generate list of streak images and patterns

y_all = {};

for k=1:N

	x=Data(k,:,:,:);
    x=squeeze(x);
    
	p=4;                                 %%%%%%%%%%%%%% binned piexl size (in fact we need use a rand mask pattern with a binned pixel size, which is related to the spatial resolution of optical system)
	C_transiton=zeros(p*size(C,1),p*size(C,2));
	for i=1:size(C,1)
	    for j=1:size(C,2)
            C_transiton(1+p*(i-1):p*i,1+p*(j-1):p*j)=C(i,j);
	    end
	end
	C=im2bw(C_transiton(1:D_x,1:D_y),0.1);
	%figure;imshow(C,[]); %%%%%%%%%%%%%%%% for inspecting 

	C_1=zeros(D_x+D_t-1,D_y,D_t);      %%%%%%%%%%%%%%% shearing the encoding mask  C            

	for i=1:D_t            
	    C_1(i:i-1+D_x,:,i)=C;
		%figure(10);imshow(C_1(:,:,i),[]);  %%% for inspecting
	end
	Cu=C_1;


	x_1=zeros(D_x+D_t-1,D_y,D_t);      %%%%%%%%%%%%%%% shearing the video x            

	for i=1:D_t            
	    x_1(i:i-1+D_x,:,i)=x(:,:,i);
		%figure(10);imshow(x_1(:,:,i),[]);  %%% for inspecting
    end
    
	x=x_1;

	y1=x.*Cu;                        %%%%%%%%%%%%%%%%%%% encoding the video with the shearing operator

	y = sum(y1,3);                   %%%%%%%%%%%%%%%%%%% integrate to obtain a streaking image


    %% Store
    y_all{k} = y;
    
end

y = cat(3,y_all{:});
y = permute(y,[3,2,1]);
mean_y = mean(reshape(y,[numel(y),1]));
std_y = std(reshape(y,[numel(y),1]));
y = (y-mean_y)/std_y;

%% Save y
h5create(output_file_name,'/input_samples',size(y))
h5write(output_file_name,'/input_samples',y);
