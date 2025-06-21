clc; clear; close all;


fc = 1e9;
c = physconst('LightSpeed');
lambda = c / fc;
M = 8;
Nsamples = 1000;
Nexamples = 2000;                     % Smaller test set
SNR_dB = 10;                          % Match or vary from training
d = lambda/2;



%% Import Test Dataset of Covariance Matrices
load('DOA_Test_Data_2deg.mat');  % loads covSet, labelSet, angleGrid

N = size(labelSet, 2);          % number of examples
R = cell(1, N);                 % preallocate cell array for matrices

for i = 1:N
    R_real = squeeze(covSet(:, :, i));          % real part
    R_imag = squeeze(covSet(:, :, i + N));      % imag part
    R{i} = R_real + 1j * R_imag;                % combine
end

test_data = 100;
R_test = R{test_data};
K = nnz(labelSet(:, test_data));  % Number of non-zero entries = number of DOAs
labelSet(:, test_data)

%% Generate Struct
DOA_data = struct();

for i = 1:N
    R_real = squeeze(covSet(:, :, i));
    R_imag = squeeze(covSet(:, :, i + N));
    DOA_data(i).R = R_real + 1j * R_imag;
    DOA_data(i).numSources = nnz(labelSet(:, i));
    DOA_data(i).labelBins = find(labelSet(:, i));  % optional: get active bins
end

%% MUSIC Part

sample = 100;

% --- Array Parameters ---
M = 8;                          % Number of antennas
d = 0.5;                        % Inter-element spacing in wavelengths
lambda = 1;                    % Wavelength
angle_scan = -90:1:90;         % Scan range in degrees
num_sources = DOA_data(sample).numSources;   % Number of sources for snapshot 1

% --- Steering Vector Function ---
steerVec = @(theta) exp(1j * 2 * pi * d * (0:M-1)' * sind(theta));  % M×1 steering vector

% --- Eigen-decomposition of Covariance Matrix ---
R1 = R{sample};                     % Covariance matrix for snapshot 1
[Evec, Eval] = eig(R1);
[evals_sorted, idx] = sort(diag(Eval), 'ascend');
En = Evec(:, idx(1:end-num_sources));  % Noise subspace

% --- MUSIC Spectrum Computation ---
music_spectrum = zeros(size(angle_scan));

for k = 1:length(angle_scan)
    a = steerVec(angle_scan(k));  % M×1
    music_spectrum(k) = 1 / (a' * (En * En') * a);
end

% --- Normalize and Plot Spectrum ---
music_spectrum_dB = 10*log10(abs(music_spectrum));
music_spectrum_dB = music_spectrum_dB - max(music_spectrum_dB);  % normalize

figure;
plot(angle_scan, music_spectrum_dB, 'LineWidth', 2);
xlabel('Angle (°)');
ylabel('Spatial Spectrum (dB)');
title('MUSIC Spectrum for R{1}');
grid on;

% --- Find Peaks (DOAs) ---
[~, peak_indices] = maxk(music_spectrum, num_sources);
estimated_DOAs = angle_scan(peak_indices);
disp('Estimated DOAs (degrees):');
disp(sort(estimated_DOAs));


% %% === STEP 2: Eigen-Decomposition ===
% [U, D] = eig(R_test);
% [eigvals, idx] = sort(diag(D), 'descend');
% U = U(:, idx);  % Sort eigenvectors accordingly
% 
% % Separate signal and noise subspaces
% Us = U(:, 1:K);          % Signal subspace
% Un = U(:, K+1:end);      % Noise subspace
% 
% %% === STEP 3: MUSIC Spatial Spectrum ===
% scanAngles = -90:0.5:90;
% Pmusic = zeros(size(scanAngles));
% a = @(theta) exp(1j * 2 * pi * d * sind(theta) / lambda * (0:M-1).'); % Mx1 steering vector
% 
% for i = 1:length(scanAngles)
%     steeringVec = a(scanAngles(i));
%     Pmusic(i) = 1 / (steeringVec' * (Un * Un') * steeringVec);
% end
% 
% % Convert to dB scale
% Pmusic_dB = 10 * log10(abs(Pmusic) / max(abs(Pmusic)));
% 
% %% === STEP 4: Plot MUSIC Spectrum ===
% figure;
% plot(scanAngles, Pmusic_dB, 'b', 'LineWidth', 2); hold on;
% % xline(trueAngles(1), 'k--', 'LineWidth', 1.5);
% % xline(trueAngles(2), 'k--', 'LineWidth', 1.5);
% xlabel('Angle (degrees)');
% ylabel('Pseudo-Spectrum (dB)');
% title('MUSIC DOA Estimation');
% legend('MUSIC Spectrum', 'True DOAs');
% grid on;
% xlim([-90, 90]);
