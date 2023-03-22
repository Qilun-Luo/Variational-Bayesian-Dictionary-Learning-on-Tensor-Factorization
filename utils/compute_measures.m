%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
%FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
%DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
%SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
%CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
%OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Nil Goyette
% University of Sherbrooke
% Sherbrooke, Quebec, Canada. April 2012

function [confusionMatrix, stats] = compute_measures(videoPath, binaryFolder, range, extension)
    % A video folder should contain 2 folders ['input', 'groundtruth']
	% and the "temporalROI.txt" file to be valid. The choosen method will be
	% applied to all the frames specified in \temporalROI.txt
    
    idxFrom = range(1);
    idxTo = range(2);
%     inputFolder = fullfile(videoPath, 'input');
   % disp(['Comparing ', videoPath, ' with ', inputFolder, newline, 'From frame ' ,  num2str(idxFrom), ' to ',  num2str(idxTo), newline]);

    % Compare your images with the groundtruth and compile statistics
    groundtruthFolder = fullfile(videoPath, 'groundtruth');
    confusionMatrix = compareImageFiles(groundtruthFolder, binaryFolder, idxFrom, idxTo, extension);
    [confusionMatrix, stats] = confusionMatrixToVar(confusionMatrix);
    
end



function confusionMatrix = compareImageFiles(gtFolder, binaryFolder, idxFrom, idxTo, extension)
    % Compare the binary files with the groundtruth files.
    
%     extension = '.jpg'; % TODO Change extension if required
    threshold = strcmp(extension, '.jpg') == 1 || strcmp(extension, '.jpeg') == 1;
    
    imBinary = imread(fullfile(binaryFolder, ['bin', num2str(idxFrom, '%.6d'), extension]));
    int8trap = isa(imBinary, 'uint8') && min(min(imBinary)) == 0 && max(max(imBinary)) == 1;
    
    confusionMatrix = [0 0 0 0 0]; % TP FP FN TN SE
    for idx = idxFrom:idxTo
        fileName = num2str(idx, '%.6d');
        imBinary = imread(fullfile(binaryFolder, ['bin', fileName, extension]));
        if size(imBinary, 3) > 1
            imBinary = rgb2gray(imBinary);
        end
        if islogical(imBinary) || int8trap
            imBinary = uint8(imBinary)*255;
        end
        if threshold
            imBinary = im2bw(imBinary, 0.5);
            imBinary = im2uint8(imBinary);
        end
        imGT = imread(fullfile(gtFolder, ['gt', fileName, '.png']));
        
        confusionMatrix = confusionMatrix + compare(imBinary, imGT);
    end
end

function confusionMatrix = compare(imBinary, imGT)
    % Compares a binary frames with the groundtruth frame
    
    TP = sum(sum(imGT==255&imBinary==255));		% True Positive 
    TN = sum(sum(imGT<=50&imBinary==0));		% True Negative
    FP = sum(sum((imGT<=50)&imBinary==255));	% False Positive
    FN = sum(sum(imGT==255&imBinary==0));		% False Negative
    SE = sum(sum(imGT==50&imBinary==255));		% Shadow Error
	
    confusionMatrix = [TP FP FN TN SE];
end

function [confusionMatrix, stats] = confusionMatrixToVar(confusionMatrix)
    TP = confusionMatrix(1);
    FP = confusionMatrix(2);
    FN = confusionMatrix(3);
    TN = confusionMatrix(4);
    SE = confusionMatrix(5);
    
    recall = TP / (TP + FN);
    specficity = TN / (TN + FP);
    FPR = FP / (FP + TN);
    FNR = FN / (TP + FN);
    PBC = 100.0 * (FN + FP) / (TP + FP + FN + TN);
    precision = TP / (TP + FP);
    FMeasure = 2.0 * (recall * precision) / (recall + precision);
    
    stats = [recall specficity FPR FNR PBC precision FMeasure];
end
