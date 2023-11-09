import java.sql.Time
import java.text.SimpleDateFormat
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.*

//val input =
//    "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" /><title>QQ Message</title><style type=\"text/css\">body{font-size:12px; line-height:22px; margin:2px;}td{font-size:12px; line-height:22px;}</style></head><body><table width=100% cellspacing=0><tr><td><div style=padding-left:10px;><br><b>消息记录</b></div></td></tr><tr><td><div style=padding-left:10px;>消息分组:我加入的群聊</div></td></tr><tr><td><div style=padding-left:10px;>消息对象:咕咕瓜的避难窝</div></td></tr><tr><td><div style=padding-left:10px;>&nbsp;</div></td></tr><tr><td style=border-bottom-width:1px;border-bottom-color:#8EC3EB;border-bottom-style:solid;color:#3568BB;font-weight:bold;height:24px;line-height:24px;padding-left:10px;margin-bottom:5px;>日期: 2020-02-11</td></tr><tr><td><div style=color:#006EFE;padding-left:10px;><div style=float:left;margin-right:6px;>系统消息(10000)</div>4:36:46</div><div style=padding-left:20px;><font style=\"font-size:9pt;font-family:'','MS Sans Serif',sans-serif;\" color='505050'>你已经是群成员了，和大家打个招呼吧！</font></div></td></tr>"
//val DATE_REGEX = Regex(".*日期: (\\d{4}-\\d{2}-\\d{2}).*")
//val YYYY_MM_DD_DATE_FORMATTER = SimpleDateFormat("yyyy-MM-dd").apply { timeZone = TimeZone.getTimeZone("UTC") }
//val H_MM_SS_DATETIMEFORMATTER = DateTimeFormatter.ofPattern("H:mm:ss")
//val systemDefaultTimeZone = TimeZone.getDefault()
//val time = "22:32:21"
//
//System.err.println(DATE_REGEX.matches(input))
//val yyyyMmDd = DATE_REGEX.find(input)!!.groupValues[1]
//val parsedInstant = YYYY_MM_DD_DATE_FORMATTER.parse(yyyyMmDd).toInstant()
//System.err.println(parsedInstant.toString())
//val localTime = LocalTime.parse(time, H_MM_SS_DATETIMEFORMATTER)
//System.err.println("${localTime.hour}:${localTime.minute}:${localTime.second}")
//val calIns = Calendar.getInstance()
//calIns.time = Time.from(parsedInstant)
//calIns.timeZone = systemDefaultTimeZone
//calIns.set(Calendar.HOUR_OF_DAY, localTime.hour)
//calIns.set(Calendar.MINUTE, localTime.minute)
//calIns.set(Calendar.SECOND, localTime.second)
//System.err.println(calIns.time)


var df: SimpleDateFormat = SimpleDateFormat("h:mm:ss aa", Locale.ENGLISH);
val text = df.format(Date(1664640046000L))
System.out.println("Time - " + text);