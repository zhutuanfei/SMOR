function synSamples = SMOR(data,targets,consid_L,syn_n,k,w,r)
%--------------------------------------------------------------------------
%SMOR: The implementation for the oversampling algorithm SMOR

%Input:
%data:The original dataset.NOTE THAT THE INPUT DATA SHOULD BE
%STANDARDIZED IN ADVANCE
%targets:The labels 
%consid_L:The considered minority class
%syn_n: The number of synthetic samples that need to be generated
%k: Number of nearest neighbors considered to construct candidate assistant seed samples set
%w and r: The parameters used in the calculation of selection weights 

%Output:
%synSamples: The generated synthetic samples set for class consid_L
%-----------------------------------------------------------------------end
if nargin < 4
   error(message('the number of the input paramters is wrong!'));
end

if nargin < 5
   k=8;
   w=0.25;
   r=1/4;
elseif nargin < 6
   w=0.25;
   r=1/4;
elseif nargin < 7
   r=1/4;
end

if syn_n==0||isempty(data)
   disp('warnning:the number of synthetic samples to be generated is 0 or the input dataset is empty!')
   synSamples=[];
   return;
end

previous_ind = find(targets == (consid_L - 1))';
cur_ind = find(targets == consid_L)';
next_ind = find(targets == (consid_L + 1))';

curn = numel(cur_ind);
pren = numel(previous_ind);
nextn = numel(next_ind);

if curn==0
   disp('warnning:the minroity class under consideration is not found in the input dataset!')
   synSamples=[];
   return;
end

CAS = cell(curn,1); %each cell save the indexs of candidate assistant seed samples
D = pdist2(data(cur_ind,:),data,'sqeuclidean');  %compute the distances between samples

%----------------------construction of candidate assistant seed samples set           
if pren~=0,   % Is the considered class the first class?
   D1 = D(:,previous_ind);
   D2 = D1;
   adjacency_1 = zeros(curn,pren);
   adjacency_2 = zeros(curn,pren);
       
   for i=1:curn,
      for j=1:k,
          [M,idx] = min(D1(i,:));
           D1(i,idx) = inf;
           adjacency_1(i,idx) = 1;
      end
   end
        
   neighbors = sum(adjacency_1,1);
   ind = find(neighbors>0);
   for i=ind,
      for j=1:k,
          [M,idx] = min(D2(:,i));
          D2(idx,i) = inf;
          adjacency_2(idx,i) = 1;
       end
    end
        
    adjacency_G1 = adjacency_1 .* adjacency_2;
    for i=1:curn
        CAS{i} = [CAS{i} previous_ind(adjacency_G1(i,:)>0)];
    end
 end
    
 if nextn~=0,     % Is the considered class the last class?
     D1 = D(:,next_ind);
     D2 = D1;
     adjacency_1 = zeros(curn,nextn);
     adjacency_2 = zeros(curn,nextn);
       
     for i=1:curn,
         for j=1:k,
             [M,idx] = min(D1(i,:));
              D1(i,idx) = inf;
              adjacency_1(i,idx) = 1;
         end
      end
        
      neighbors = sum(adjacency_1,1);
      ind = find(neighbors>0);
   for i=ind,
       for j=1:k,
           [M,idx] = min(D2(:,i));
           D2(idx,i) = inf;
           adjacency_2(idx,i) = 1;
        end
    end
        
    adjacency_G1 = adjacency_1 .* adjacency_2; 
    for i=1:curn
        CAS{i} = [CAS{i} next_ind(adjacency_G1(i,:)>0)];
    end
 end
    
adjacency_G2 = zeros(curn,curn);
D1 = D(:,cur_ind);
for i=1:curn,
   for j=1:k+1,                        % why is k+1, because including itself in the computation of k-nearest neighbors
       [M,idx] = min(D1(i,:));
       D1(i,idx) = inf;
       adjacency_G2(i,idx) = 1;
    end
end

adjacency_G2 = adjacency_G2 - eye(size(adjacency_G2)); 
for i=1:curn
    CAS{i} = [CAS{i} cur_ind(adjacency_G2(i,:)>0)];
end

clear adjacency_G2 D1 D2 adjacency_G1 adjacency_1  adjacency_2
%-----------------------------------------------------------------------end  

%-------------------------------------------performing the first filtration
for i=1:curn
    D(i,cur_ind(i))=Inf;
end
firstFiltration = cell(curn,2); 
for i=1:curn
    r_max = max(D(i,CAS{i}));
    firstFiltration{i,1}=find(D(i,:)<=r_max);
    firstFiltration{i,2}=D(i,firstFiltration{i,1});
end
%-----------------------------------------------------------------------end


%----------------------------------the computation of the selection weights
SW=cell(curn,1);         %the selection weights, each cell represents a vector of selection weight
for i=1:curn
    ind_i = cur_ind(i);
    for j=1:numel(CAS{i})
        ind_j=CAS{i}(j);
        ind_j_trap = find(cur_ind==ind_j);%if ind_ij belongs to the index of the  trapped samples
        if ~isempty(ind_j_trap)
           l=find(CAS{ind_j_trap}==ind_i);
           if ~isempty(l)&&~isempty(SW{ind_j_trap})
              SW{i}(j) = SW{ind_j_trap}(l);
              continue;
           end 
        end
        SW{i}(j)=IPA(data,ind_i,ind_j,firstFiltration{i,1},firstFiltration{i,2},targets,consid_L,w,r);
    end
end
%-----------------------------------------------------------------------end


%---------------------------------------the allocation of synthetic samples
og_n = floor(syn_n/curn).*ones(curn,1);
Ind = randsample(1:curn,syn_n-curn*floor(syn_n/curn),false);
og_n(Ind)=og_n(Ind)+1;
%-----------------------------------------------------------------------end


%----------------------------------------the generation of synthetic samples
g_count=0;
synSamples = zeros(syn_n,size(data,2));
for i=1:curn
    ind_i = cur_ind(i);
    if og_n(i)>0
       synSamples(g_count+1:g_count+og_n(i),:) = interpolaGenerat(data,targets,ind_i,CAS{i},SW{i},og_n(i));   
       g_count=g_count+og_n(i);
    end
end
%-----------------------------------------------------------------------end
end

function SI = interpolaGenerat(samples,targets,OSind,cas,sw,syn_n)
w1=0.25;
if syn_n<=0 || isempty(samples)
    SI=[];
    return;
end

if numel(sw)~=numel(cas)
   error('size of selection weight vector is not equal to the number of candidiate assistant seed samples!');
end

if numel(find(sw>=1))==0
   cas = [OSind cas];
   sw = [1+w1*exp(-1) sw];
end

rni = randsample(numel(cas),syn_n,true,sw./sum(sw)); 

rni = cas(rni);

adj_class_ind = find(targets(rni)~=targets(OSind));

gap = rand(syn_n,size(samples,2));

gap(adj_class_ind,:) = gap(adj_class_ind,:)./2;

SI=repmat(samples(OSind,:),syn_n,1)+gap.*(samples(rni,:)-repmat(samples(OSind,:),syn_n,1));
end


function sw=IPA(data, PSSind,ASSind,FSInd,FSDis,targets,consid_L,w,r)
%Identifying the samples of PA neighborhood
midPoint = (data(ASSind,:)+data(PSSind,:))./2;
dis_PRImid= pdist2(data(PSSind,:),midPoint,'sqeuclidean');
dis_PRIARS= pdist2(data(PSSind,:),data(ASSind,:),'sqeuclidean');
secondFiltration = FSInd(roundn(FSDis,-5)<=roundn(dis_PRIARS,-5));
secondFiltration = reshape(secondFiltration,1,numel(secondFiltration));
tempCnInd=[];
for l=secondFiltration
    dis_midCur=pdist2(midPoint,data(l,:),'sqeuclidean');
    if roundn(dis_midCur,-5)<=roundn(dis_PRImid,-5)
       tempCnInd=[tempCnInd;l];
    end
end
curClassCount = numel(find(targets(tempCnInd)==consid_L))+1;
otherClassCount = numel(tempCnInd)-curClassCount+1;
difficutlyfactor = w*(1-exp(-r*(otherClassCount/curClassCount)));
classes = unique(targets);classes(classes==consid_L)=[];
class_size=zeros(1,numel(classes));
for i=1:numel(classes)
    class_size(i)=numel(find(targets(tempCnInd)==classes(i)));
end
Fr = r*sum(abs(consid_L-classes)'.*(class_size./curClassCount));
sw = exp(-Fr)+difficutlyfactor;         
end



function D = pdist2( X, Y, metric )
%NOTE: This function is providied by the paper entitled 
%      "Graph-Based Approaches for Over-sampling in the context of Ordinal Regression"
% Calculates the distance between sets of vectors.
%
% Let X be an m-by-p matrix representing m points in p-dimensional space
% and Y be an n-by-p matrix representing another set of points in the same
% space. This function computes the m-by-n distance matrix D where D(i,j)
% is the distance between X(i,:) and Y(j,:).  This function has been
% optimized where possible, with most of the distance computations
% requiring few or no loops.
%
% The metric can be one of the following:
%
% 'euclidean' / 'sqeuclidean':
%   Euclidean / SQUARED Euclidean distance.  Note that 'sqeuclidean'
%   is significantly faster.
%
% 'chisq'
%   The chi-squared distance between two vectors is defined as:
%    d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2;
%   The chi-squared distance is useful when comparing histograms.
%
% 'cosine'
%   Distance is defined as the cosine of the angle between two vectors.
%
% 'emd'
%   Earth Mover's Distance (EMD) between positive vectors (histograms).
%   Note for 1D, with all histograms having equal weight, there is a simple
%   closed form for the calculation of the EMD.  The EMD between histograms
%   x and y is given by the sum(abs(cdf(x)-cdf(y))), where cdf is the
%   cumulative distribution function (computed simply by cumsum).
%
% 'L1'
%   The L1 distance between two vectors is defined as:  sum(abs(x-y));
%
%
% USAGE
%  D = pdist2( X, Y, [metric] )
%
% INPUTS
%  X        - [m x p] matrix of m p-dimensional vectors
%  Y        - [n x p] matrix of n p-dimensional vectors
%  metric   - ['sqeuclidean'], 'chisq', 'cosine', 'emd', 'euclidean', 'L1'
%
% OUTPUTS
%  D        - [m x n] distance matrix
%
% EXAMPLE
%  [X,IDX] = demoGenData(100,0,5,4,10,2,0);
%  D = pdist2( X, X, 'sqeuclidean' );
%  distMatrixShow( D, IDX );
%
% See also PDIST, DISTMATRIXSHOW

% Piotr's Image&Video Toolbox      Version 2.0
% Copyright (C) 2007 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Lesser GPL [see external/lgpl.txt]
if( nargin<3 || isempty(metric) ); metric=0; end;

switch metric
  case {0,'sqeuclidean'}
    D = distEucSq( X, Y );
  case 'euclidean'
    D = sqrt(distEucSq( X, Y ));
  case 'L1'    
    D = distL1( X, Y );
  case 'cosine'
    D = distCosine( X, Y );
  case 'emd'
    D = distEmd( X, Y );
  case 'chisq'
    D = distChiSq( X, Y );
  otherwise
    error(['pdist2 - unknown metric: ' metric]);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = distL1( X, Y )

m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yi = yi( mOnes, : );
  D(:,i) = sum( abs( X-yi),2 );
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = distCosine( X, Y )

if( ~isa(X,'double') || ~isa(Y,'double'))
  error( 'Inputs must be of type double'); end;

p=size(X,2);
XX = sqrt(sum(X.*X,2)); X = X ./ XX(:,ones(1,p));
YY = sqrt(sum(Y.*Y,2)); Y = Y ./ YY(:,ones(1,p));
D = 1 - X*Y';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = distEmd( X, Y )
Xcdf = cumsum(X,2);
Ycdf = cumsum(Y,2);

m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  ycdf = Ycdf(i,:);
  ycdfRep = ycdf( mOnes, : );
  D(:,i) = sum(abs(Xcdf - ycdfRep),2);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = distChiSq( X, Y )

%%% supposedly it's possible to implement this without a loop!
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yiRep = yi( mOnes, : );
  s = yiRep + X;    d = yiRep - X;
  D(:,i) = sum( d.^2 ./ (s+eps), 2 );
end
D = D/2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = distEucSq( X, Y )

%if( ~isa(X,'double') || ~isa(Y,'double'))
 % error( 'Inputs must be of type double'); end;
m = size(X,1); n = size(Y,1);
%Yt = Y';
XX = sum(X.*X,2);
YY = sum(Y'.*Y',1);
D = XX(:,ones(1,n)) + YY(ones(1,m),:) - 2*X*Y';
end
