%Exercise 1
% Read the image
Im = imread('barbara.jpg');
% Define the levels
L = [8, 12, 16, 20];
% Show original image
figure;
subplot(3, 2, 1);
imshow(Im);
title('Original Image');
axis off;
% Loop over levels
for i = 1:length(L)
    Levels = L(i);
    % Apply quantization
    Q = midrise(Im, Levels);
    % Show quantized image
    subplot(3, 2, i + 1);
    imshow(Q);
    % Compute the mean squared error
    mse_value = mse(Im, Q);
    title(sprintf('Levels: %d ,MSE: %.2f', Levels, mse_value));
    axis off;
end

%Exercise 2
%Read the image
Im=imread('barbara.jpg');
Im_double=im2double(Im);
% Compute the 2D Discrete Fourier Transform (DFT)
DFT_Im = fft2(Im_double);
% Fasma Platous
Fasma_Platous = abs(DFT_Im);
% Fasma Fashs
Fasma_Fashs = angle(DFT_Im);
% Display the original image
figure;
imshow(Im_double);
title('Original Image');
axis off;
% Display the fasma platous
imshow(log(1 + Fasma_Platous), []);
title('Fasma Platous');
axis off;
% Display the fasma fashs
imshow(Fasma_Fashs, []);
title('Fasma Fashs');
axis off;

%For the calculation of the inverse Fourier transform
% Read the image
Im = imread('barbara.jpg');
Im_double = double(Im);
% Compute the 2D Discrete Fourier Transform (DFT)
DFT_Im = fft2(Im_double);
% Fasma Platous
Fasma_Platous = abs(DFT_Im);
sorted_Fasma_Platous = sort(Fasma_Platous(:), 'descend');
% Adjust oriakes times
top_20 = sorted_Fasma_Platous(round(0.2 * numel(DFT_Im)));
top_40 = sorted_Fasma_Platous(round(0.4 * numel(DFT_Im)));
DFT_Im_20 = DFT_Im .* (abs(DFT_Im) >= top_20);
DFT_Im_40 = DFT_Im .* (abs(DFT_Im) >= top_40);
% Reconstruct the images using the inverse Fourier transform
Im_recon_20 = ifft2(DFT_Im_20);
Im_recon_40 = ifft2(DFT_Im_40);
% Ensure the real part
Im_recon_20 = real(Im_recon_20);
Im_recon_40 = real(Im_recon_40);
% Show the reconstructed images
figure;
imshow(uint8(abs(Im_recon_20)));
title('Reconstructed Image (20%)');
imshow(uint8(abs(Im_recon_40)));
title('Reconstructed Image (40%)');
% Calculate the mean squared error (MSE) for each reconstruction
mse_20 = immse(Im_double, abs(Im_recon_20));
mse_40 = immse(Im_double, abs(Im_recon_40));
% Print the MSE values
fprintf('MSE for 20%% coefficients: %.4f\n', mse_20);
fprintf('MSE for 40%% coefficients: %.4f\n', mse_40);

%Exercise 3
%For printing the dct and rdct images
% Load the grayscale image
Im = imread('barbara.jpg');
Im_double = double(Im);
% Calculate the forward 2D DCT
DCT_Im = DCT(Im_double);
figure;
imshow(uint8(abs(Im_double)));
title('DCT image');
DCT_R_Im=RDCT(Im_double);
figure;
imshow(uint8(abs(Im_double)));
title('RDCT image');

%For printing platos 
% Load the grayscale image
Im = imread('barbara.jpg');
Im_double = double(Im);
% Calculate 2D DCT
DCT_Im = DCT(Im_double);
figure;
imshow(uint8(abs(Im_double)));
title('DCT image');
% Calculate the spectrum of the transformation
Platos = abs(DCT_Im);
% Print Platos
figure;
imagesc(log(1 + Platos));
colorbar;
title('DCT Platos');

%Exercise 4
%Read the image
Im = imread('lenna.jpg');
%printing original image
imshow(Im);
title('Αρχική Εικόνα');
%applying salt & pepper
noisy_Im = imnoise(Im, 'salt & pepper', 0.05);
%printing noisy image
imshow(noisy_Im);
title('Εικόνα με Θόρυβο');

%for 3x3 filter
filter_3x3=medfilt2 (noisy_Im, [3 3]);
figure;
imshow(filter_3x3);
title('Filtrarismeni eikona 3x3');

%for 5x5 filter
filter_5x5=medfilt2 (noisy_Im, [5 5]);
figure;
imshow(filter_5x5);
title('Filtrarismeni eikona 5x5');

%for 7x7 filter
filter_7x7=medfilt2 (noisy_Im, [7 7]);
figure;
imshow(filter_7x7);
title('Filtrarismeni eikona 7x7');

%Exercise 5
% Load the image
Im=imread('lenna.jpg');
% Print the original image
imshow(Im);
title('Original Image');
% Add Gaussian noise
noisy_Im = imnoise(Im, 'gaussian', 0, 0.01);
% Display the image with Gaussian noise
imshow(noisy_Im);
title('Image with Gaussian Noise');

%for 3h taksh
Im=imread('noisy_Im');
figure;
imshow(noisy_Im)
title('Arxiki eikona');
Do=0.2; %sixnotita apokopis
order=3; %Taksi filtrou
Do=255*Do;
[M,N]=size(noisy_Im);
[x,y]=meshgrid(-floor(N/2):floor((N-1)/2),-floor(M/2):floor((M-1)/2));
Fasma_But= sqrt (1./(1+((x.^2+y.^2)/Do^2).^order));%Fasma filtrou
%Filtrarisma eikonas
Fasma_Im = fftshift(fft2(double(noisy_Im)));
Fasma = Fasma_But.* Fasma_Im;
%Prosdiorismos telikis eikonas
Im_3taksi=uint8( abs(ifft2(ifftshift(Fasma))));
figure;
imshow(Im_3taksi);
title('Filtrarismenh eikona 3hs takshs');

%for 5h taksh
Im=imread('noisy_Im');
figure;
imshow(noisy_Im)
title('Arxiki eikona');
Do=0.2; %sixnotita apokopis
order=5; %Taksi filtrou
Do=255*Do;
[M,N]=size(noisy_Im);
[x,y]=meshgrid(-floor(N/2):floor((N-1)/2),-floor(M/2):floor((M-1)/2));
Fasma_But= sqrt (1./(1+((x.^2+y.^2)/Do^2).^order));%Fasma filtrou
%Filtrarisma eikonas
Fasma_Im = fftshift(fft2(double(noisy_Im)));
Fasma = Fasma_But.* Fasma_Im;
%Prosdiorismos telikis eikonas
Im_5taksi=uint8( abs(ifft2(ifftshift(Fasma))));
figure;
imshow(Im_5taksi);
title('Filtrarismenh eikona 5hs takshs');

%for 7h taksh
Im=imread('noisy_Im');
figure;
imshow(noisy_Im)
title('Arxiki eikona');
Do=0.2; %sixnotita apokopis
order=7; %Taksi filtrou
Do=255*Do;
[M,N]=size(noisy_Im);
[x,y]=meshgrid(-floor(N/2):floor((N-1)/2),-floor(M/2):floor((M-1)/2));
Fasma_But= sqrt (1./(1+((x.^2+y.^2)/Do^2).^order));%Fasma filtrou
%Filtrarisma eikonas
Fasma_Im = fftshift(fft2(double(noisy_Im)));
Fasma = Fasma_But.* Fasma_Im;
%Prosdiorismos telikis eikonas
Im_7taksi=uint8( abs(ifft2(ifftshift(Fasma))));
figure;
imshow(Im_7taksi);
title('Filtrarismenh eikona 7hs takshs');

%Exercise 6
% Load the image
Im=imread('butterfly.jpg');
Im = double(Im);
% Μετασχηματισμός της έγχρωμης εικόνας στον χρωματικό χώρο RGB
[m, n, ~] = size(Im);
X = reshape(Im, m * n, 3);
% Define values of K
K_values = [5, 10, 15];
% Iterate over each value of K
for i = 1:length(K_values)
    K = K_values(i);
    % Randomly initialize centroids
    centroids = X(randperm(size(X, 1), K), :);
    % Set the maximum number of iterations
    T = 50;
    % Run K-means
    for iter = 1:T
        % Assign each data point to the nearest centroid
        % Compute the Euclidean distance between each data point and each centroid
        distances = zeros(size(X, 1), K);
        for k = 1:K
            distances(:, k) = sum((X - centroids(k, :)).^2, 2);
        end
        % Find the closest centroid for each data point
        [~, idx] = min(distances, [], 2);
        % Update centroids
        for k = 1:K
            centroids(k, :) = mean(X(idx == k, :));
        end

        % Check for convergence (if the centroids didn't change, break the loop)
        if iter > 1 && isequal(prev_centroids, centroids)
            break;
        end
        % Store the previous centroids for convergence check
        prev_centroids = centroids;
    end
    % Assign each pixel to its cluster centroid
    clustered_Im = zeros(m * n, 3);
    for k = 1:K
        clustered_Im(idx == k, :) = repmat(centroids(k, :), sum(idx == k), 1);
    end
    % Reshape the clustered image back to its original dimensions
    clustered_Im = reshape(clustered_Im, m, n, 3);
    % Convert the clustered image to grayscale for visualization
    grayscale_clustered_Im = rgb2gray(uint8(clustered_Im));
    % Display the segmented image
    figure;
    imshow(uint8(grayscale_clustered_Im));
    title(sprintf('Segmented Image (K = %d)', K));
    colormap(gray);
end



