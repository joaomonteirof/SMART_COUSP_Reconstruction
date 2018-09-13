	
mask=load('mask.mat');
Rm=mask.mask2;
load('test.mat');
x=x_;

[D_x,D_y,D_t]=size(x);

C=Rm;

p=4;                                 %%%%%%%%%%%%%% binned piexl size (in fact we need use a rand mask pattern with a binned pixel size, which is related to the spatial resolution of optical system)
C_transiton=zeros(p*size(C,1),p*size(C,2));
for i=1:size(C,1)
    for j=1:size(C,2)
        C_transiton(1+p*(i-1):p*i,1+p*(j-1):p*j)=C(i,j);
    end
end
C=im2bw(C_transiton(1:D_x,1:D_y),0.1);
%figure;imshow(C,[]); %%%%%%%%%%%%%%%% for inspecting 
C=Rm;
C_1=zeros(D_x+D_t-1,D_y,D_t);      %%%%%%%%%%%%%%% shearing the encoding mask  C            

for i=1:D_t            
    C_1(i:i-1+D_x,:,i)=C;
    %figure(10);imshow(C_1(:,:,i),[]);  %%% for inspecting
end
Cu=C_1;

x_1=zeros(D_x+D_t-1,D_y,D_t);      %%%%%%%%%%%%%%% shearing the video x            

for i=1:D_t            
    x_1(i:i-1+D_x,:,i)=x(:,:,i);
    figure(10);imshow(x_1(:,:,i),[]);  %%% for inspecting
end

x=x_1;

y1=x.*Cu;                        %%%%%%%%%%%%%%%%%%% encoding the video with the shearing operator

y = sum(y1,3);                   %%%%%%%%%%%%%%%%%%% integrate to obtain a streaking image