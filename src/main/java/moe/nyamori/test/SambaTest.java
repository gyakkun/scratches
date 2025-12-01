package moe.nyamori.test;

import jcifs.FileNotifyInformation;
import jcifs.smb.SmbException;
import jcifs.smb.SmbFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.MalformedURLException;
import java.util.Base64;
import java.util.Collections;

public class SambaTest {
    public static final Logger LOGGER = LoggerFactory.getLogger(SambaTest.class);

    public static void main(String[] args) throws MalformedURLException, SmbException {
        var smbFile = new SmbFile("smb://Steve:xiaowu1996@Steve-PC/torrents/");
//        for (var i : smbFile) {
//            System.err.println(i);
//        }
        var filter = FileNotifyInformation.FILE_ACTION_ADDED;
//                | FileNotifyInformation.FILE_ACTION_REMOVED
//                | FileNotifyInformation.FILE_ACTION_MODIFIED
//                | FileNotifyInformation.FILE_ACTION_RENAMED_OLD_NAME
//                | FileNotifyInformation.FILE_ACTION_RENAMED_NEW_NAME
//                | FileNotifyInformation.FILE_ACTION_REMOVED_BY_DELETE;
        try (var folder = smbFile;
             var watchHandle = folder.watch(filter, true);) {
            var files = Collections.<FileNotifyInformation>emptyList();
            while (!Thread.currentThread().isInterrupted()) {
                files = watchHandle.watch();
                for (var f : files) {
                    var fn = f.getFileName();
                    var ac = f.getAction();
                    LOGGER.info("file {} action {}", fn, ac);
                    try (var theFile = ((SmbFile) folder.resolve(fn))) {
                        var url = theFile.getURL();
                        LOGGER.info("File url {}", url);
                        var canRead = theFile.canRead();
                        LOGGER.info("can read {}", canRead);
                        var maxWait = 50;
                        var count = 0;
                        while (!(canRead = theFile.canRead()) && ++count <= maxWait) {
                            Thread.sleep(500);
                        }
                        if (!canRead) {
                            LOGGER.error("Cannot read {}", theFile.getURL());
                            continue;
                        }
                        count = 0;
                        while (++count <= maxWait) {
                            try (var ins = theFile.getInputStream()) {
                                var bytes = ins.readAllBytes();
                                var b64 = Base64.getEncoder().encodeToString(bytes);
                                var sub = b64.substring(0, Math.min(b64.length(), 5));
                                LOGGER.info("content: {}", sub);
                                break;
                            } catch (Exception ex) {
                                LOGGER.error("[{}-th] failed to read file {}", count, theFile.getURL());
                            }
                            Thread.sleep(500);
                        }

                    }
                }
            }

        } catch (Exception ex) {
            LOGGER.error("Ex ", ex);
        }
    }
}
