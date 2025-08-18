function out = mapGetOrEmpty(mapObj, keyStr)
% Safe containers.Map lookup that returns "" if key is missing.
% mapObj: containers.Map with char keys
% keyStr: string/char for the key
    key = char(string(keyStr));
    if isKey(mapObj, key)
        out = string(mapObj(key));
    else
        out = "";
    end
end
