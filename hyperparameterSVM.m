function [best_model,Errors,best_modelParameters] = hyperparameterSVM(arg1,arg2)

if nargin == 2
    % evaluation by cross-validation
    %=================================================================
    options = c2s(arg2);
    if ~isfield(options,'verb'), options.verb = 0; end
    if ~isfield(options,'solver'), options.solver = 'smo'; end
    if ~isfield(options,'ker'), options.ker = 'linear'; end
    if ~isfield(options,'num_folds'), options.num_folds = 5; end
    if ~isfield(options,'solver_options'), options.solver_options = []; end
    
    nargs = size(options.arg,2);
    nc = length(options.C);
    [dim,num_data] = size(arg1.X);
    
    %   solver_options = options.solver_options;
    %   solver_options.ker = options.ker;
    min_error = inf;
    Errors = [];
    cnt_model = 0;
    num_model = nc*nargs;
    best_modelParameters.C = 0;
    best_modelParameters.arg = 0;
    
    for i = 1:nargs
        arg = options.arg(:,i);
        for j = 1:nc
            cnt_model = cnt_model + 1;
            C = options.C(j);
            
            % svm parameters
            solver_options.C = C;
            solver_options.arg = arg;
            
            % display info
            if options.verb == 1
                fprintf('Model %d/%d: C=%f, arg=', ...
                    cnt_model,num_model, C);
                fprintf('%f ', arg);
                fprintf('\n');
            end
            
            
            
            labels = arg1.y;
            samples = arg1.X;
            template = templateSVM('KernelFunction', 'rbf', 'KernelScale', arg, 'BoxConstraint', C);
            model = fitcecoc(samples, labels, 'Learners', template,'KFold',10);
            
            %model = svmtrain(labels, samples, libsvm_options);
            %         model = feval( options.solver, trn, solver_options );
            
%             if options.verb == 1
%                 [ypred, accuracy, prob_estimates] = kfoldPredict(CVMdl);%,'-b 1');
%                 %           ypred = feval( model.fun, trn.X, model );
%                 fprintf(' err = %.4f, testing', cerror( ypred,labels));
%             end
            
            ypred = kfoldPredict(model);
            
            
            fold_error = cerror( ypred, labels);
            
            if options.verb == 1
                fprintf(' err = %.4f\n', fold_error);
            end
            
            
            % cross-validation error
            Errors(j,i) =  fold_error;
            
            if min_error > fold_error
                min_error = fold_error;
                best_model = model;
                best_modelParameters.C = C;
                best_modelParameters.arg = arg;
%                 tst1 = tst;
                %         save('bestparameter.mat','best_modelParameters');
                %         save('trnData.mat','trn');
                %         save('testData.mat','tst');
                
                if options.verb == 1
                    fprintf('cross-validation error = %.4f (best so far)\n\n', fold_error);
                end
            else
                if options.verb == 1
                    fprintf('cross-validation error = %.4f\n\n', fold_error);
                end
            end
            
        end
    end
    
    % disp info
    if options.verb == 1
        fprintf('best model: C=%f, arg=', ...
            best_modelParameters.C);
        fprintf('%f ', best_modelParameters.arg);
        fprintf('\ncross-validation error = %.4f\n', min_error);
    end
    
end

%EOF