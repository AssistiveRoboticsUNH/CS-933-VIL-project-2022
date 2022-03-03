function processROSBag

addpath(Directories.ROSBagDir)
addpath(Directories.blenderDir)
addpath(Directories.URDFDir)
Utils.makeDirIfMissing(Directories.dataDir);
Utils.makeDirIfMissing(Directories.demonstrationDir);

% load messages
files = Utils.getAllFileNames(Directories.rawDataDir)';
bagFiles = cellfun(@(e) strcmp(e(end-3:end), '.bag'), files);
assert(sum(bagFiles)==1, 'exactly one bag file in directory is required')
filename = files{bagFiles};
bag = rosbag(filename);
imagesMsgs = bag.select('Topic', {'/camera2/color/image_raw/compressed'});
cellImages = imagesMsgs.readMessages("DataFormat","struct");
joyMsgs = bag.select('Topic', {'/joy'});
cellJoy = joyMsgs.readMessages("DataFormat","struct");

cellImages = cellImages(imIndStart:end);

imMsg1 = cellImages{1};
cellImages = cellfun(@(msg) processMsgsTimes(msg,imMsg1), cellImages, 'UniformOutput',false);
cellJoy = cellfun(@(msg) processMsgsTimes(msg, imMsg1), cellJoy, 'UniformOutput',false);

% process data

trial = 0;

imInd = 1;
imTrialInds = [];

for i = 1:1:length(cellJoy)
    curTime = cellJoy{i}.time;
    if cellJoy{i}.Buttons(8)
        'start'
        imTrialInds = [];
    elseif cellJoy{i}.Buttons(9) && length(imTrialInds) > 20
        'end'
        trial = trial+1;
        trialDir = [Directories.demonstrationDir '/trial' num2str(trial)];
        Utils.makeDirIfMissing(trialDir);
        % process images
        for j = 1:length(imTrialInds)
            imMsg = cellImages{imTrialInds(j)};
            im = rosReadImage(imMsg);
            scale = 4*.25;
            im = imresize(im, scale);
            imwrite(im, [trialDir '/' num2str(j) '.png'])
        end

        continue
    end

    while cellImages{imInd}.time < curTime
        imInd = imInd + 1;
    end
    imTrialInds = cat(1,imTrialInds,imInd);
   
end

end
