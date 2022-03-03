function msg = processMsgsTimes(msg, msg1)
msg.time = (double(msg.Header.Stamp.Sec) + double(msg.Header.Stamp.Nsec)/1e9) -...
    (double(msg1.Header.Stamp.Sec) + double(msg1.Header.Stamp.Nsec)/1e9); 
end