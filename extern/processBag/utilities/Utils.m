classdef Utils
    methods(Static)
        function allValidFiles = getAllFileNames(varargin)
            % dir
            dir = varargin{1};
            nameValeStruct = struct();
            if length(varargin) > 1
                if mod(length(varargin),2) == 0
                    error('must use name value pairs')
                end
                nameValeStruct = struct(varargin{2:end});
            end
            files = ls(dir);
            if ispc
                files(:, size(files,2) +1) = ' ';
                files = split(reshape(files',1,[]));
            else
                files = split(files);
            end
            allValidFiles  = cell(length(files), 1);
            ind = 0;
            for ii = 1:length(files)
                fileName = files{ii};
                if strcmp(fileName, '.') || strcmp(fileName, '..') || isempty(fileName)
                    continue
                end
                if isfield(nameValeStruct, 'contains') && ~contains(fileName, nameValeStruct.contains)
                    continue
                end
                if isfield(nameValeStruct, 'exclude') && contains(fileName, nameValeStruct.exclude)
                    continue
                end
                if isfield(nameValeStruct, 'filter') && nameValeStruct.filter(fileName)
                    continue
                end
                file = [dir '/' fileName];
                ind = ind + 1;
                allValidFiles{ind} = strip(file);
            end
            array = zeros(ind,1);
            for i = 1:ind
                fileName = Utils.getFileName(allValidFiles{i});
                firstDigit = find(isstrprop(fileName,'digit'),1);
                lastDigit = max(find(isstrprop(fileName,'digit'),99999));
                array(i) = str2double(fileName(firstDigit:lastDigit));
            end
            [~,idx] = sort(array);
            allValidFiles = allValidFiles(idx, :);
        end

        function bool = findFile(dir, fileName)
            allValidFiles = Utils.getAllFileNames(dir);
            cellFound = cellfun(@(x) strcmp(Utils.getFileName(x), fileName), allValidFiles, 'UniformOutput', false);
            total = 0;
            for i = 1: length(cellFound)
                total = total + cellFound{i};
            end
            bool = total > 0;
        end

        function fileName = getFileName(path)
            ind = max(find(path == '/',99999))+1;
            fileName = path(ind:end);
        end

        function extension = getExtension(filename)
            ind = max(find(filename == '.',99999));
            extension = filename(ind:end);
        end

        function flows = getFlows(path)
            load(path, 'flows');
        end

        function status = makeDirIfMissing(dir)
            missing = ~exist(dir, 'dir');
            status = 0;
            if ispc
                dir(dir=='/')='\';
            end
            if missing
                status = system(['mkdir ' dir]);
            end

        end

        function status = removeDir(dir)
            missing = ~exist(dir, 'dir');
            status = 0;
            if ~missing && ispc
                 dir(dir=='/')='\';
                system(['rmdir /s /q ' dir])
            elseif ~missing && ~ispc
                system(['rm -r ' dir])
            end
        end

        function status = removeFilesPattern(dir,pattern)
            missing = ~exist(dir, 'dir');
            status = 0;
            if ~missing && ispc
                dir(dir=='/')='\';
                system(['del ' dir '\' pattern ' /a /s']);
            elseif ~missing && ~ispc
                system(['find ' dir ' -type f -name ' pattern ' -delete'])
            end
        end

        function setROSEnvironmentalVariables
            if ispc
                [~, result] = system('ipconfig');
                expression = 'Wireless LAN adapter Wi-Fi.*IPv4 Address. . . . . . . . . . . : ([0-9]*.[0-9]*.[0-9]*.[0-9]*)';
                [ip, ~] = regexp(result,expression, 'tokens', 'match');
            else
                [~, result] = system('ifconfig');
                expression = 'wlp2s0:.*inet ([0-9]*.[0-9]*.[0-9]*.[0-9]*)';
                [ip, ~] = regexp(result,expression, 'tokens', 'match');
            end
            setenv('ROS_IP', ip{1}{1})
            setenv('ROS_MASTER_URI', ['http://' ip{1}{1} ':11311'])
        end

    end


end