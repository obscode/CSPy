from slackclient import SlackClient
from CSPlib.config import getconfig
cfg = getconfig()

def makeReportBlock(SN, filt, mag, emag, url):
   #d = dict(attachments=[
   #   dict(blocks=[
   #      dict(type='section', text=dict(type="mrkdwn",
   #            text="*NEW SN* {} {}-mag {:.3f} +/- {:.3f}".format(
   #               SN,filt,mag,emag))),
   #      dict(type='image', title=dict(type="plain_text", 
   #           text="{} {}-band".format(SN,filt), emoji=True),
   #           alt_text="difference image", image_url=url)
   #      ])])
   #return d
   d = [dict(type='section', text=dict(type="mrkdwn",
               text="*NEW SN* {} {}-mag {:.3f} +/- {:.3f}".format(
                  SN,filt,mag,emag))),
         dict(type='image', title=dict(type="plain_text", 
              text="{} {}-band".format(SN,filt), emoji=True),
              alt_text="difference image", image_url=url)
         ]
   return d

reportTemplate = '''{
   "attachments": [
      {
         "blocks": [
            {
               "type": "section",
               "text": {
                  "type": "mrkdwn",
                  "text": "*NEW SN* {SN} {filt}-mag {mag:.3f} +/- {emag:.3f}"
               }
            },
            {
               "type": "image",
               "title": {
                  "type": "plain_text",
                  "text": "{SN} {filt}-band",
                  "emoji": true
               },
               "image_url": "{imageURL}",
               "alt_text": "difference image"
            }
         ]
      }
   ]
}'''

def sendSimpleMessage(sc, channel, message):
    res = sc.api_call(
        "chat.postMessage",
        username = "SwopeBot",
        channel = channel,
        link_names = 1,
        text = message)
    return result

def sendReportMessage(sc, channel, SN, filt, mag, emag, imageURL):

   payload = makeReportBlock(SN, filt, mag, emag, imageURL)
   res = sc.api_call("chat.postMessage",
         username="SwopeBot",
         channel=channel,
         link_names=1, blocks=payload)
   return res

def getConnection():
   return SlackClient(cfg.data.SlackToken)

