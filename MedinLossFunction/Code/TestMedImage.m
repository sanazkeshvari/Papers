clear all
%I = imread('pout.tif');

%I = imread('cameraman.tif');
I=rgb2gray( imread('football.jpg'));
%imwrite(I,"F:\2 Median Loss\code\Images\Football.png")
%[I,map] = imread('trees.tif');

%figure(7);imshow(I)
%I = imread('eight.tif');

  J = imnoise(I,'salt & pepper',0.2);
  %imwrite(J,"F:\2 Median Loss\code\Images\Football_Noisy.png")
  %J = imnoise(J,'poisson');%'gaussian',0,0.01);
%J = imnoise(I,'speckle',0.071);
%J = imnoise(I,'gaussian',0.19);
W=3;

MI=uint8(255*ones(size(J,1),size(J,2)));
for i=W:size(J,1)-W-1
    for j=W:size(J,2)-W-1
        J1=double(J(i-W+1:i+W+1,j-W+1:j+W+1));
        x=J1(:);
        [mu,mur,muI]=MedianLossFcn(x);
        MI(i,j)=muI;
    end
end
[m,n]=size(J1);
K = medfilt2(J,[m,n]);

figure(1), imshow(J), figure(2), imshow(MI);figure(3);imshow(K)
    