import json
import pprint
import xml.etree.ElementTree

pp = pprint.PrettyPrinter(indent=2)

class VideoProperties:
  
  def __init__(self, xml_repr, playlist):
    self.playlist = playlist
    for k, v in xml_repr.attrib.iteritems():
      setattr(self, k, v)
    
    self.mp4url = xml_repr.find('{urn:mpeg:dash:schema:mpd:2011}BaseURL').text
    ds = 'dynamic_streaming'
    if self.playlist[-len(ds):] == ds:
      self.quality_id = xml_repr.attrib['id'].split('_')[3]
    else:
      self.quality_id = xml_repr.attrib['id']

    self.fname = self.mp4url.split('?')[0].split('/')[-1]

    for xml_elem in xml_repr.iter('{urn:mpeg:dash:schema:mpd:2011}SegmentBase'):
      self.indexRange = xml_elem.attrib['indexRange']
      break

  def index_pair(self):
    return (self.fname, self)
  
  def __repr__(self):
    return pp.pformat(self.__dict__)


ReprXmlElement = '{urn:mpeg:dash:schema:mpd:2011}Representation'

def is_vid(xml_repr_elem):
  return xml_repr_elem.attrib['mimeType'] == 'video/mp4'

def get_video_objs(json):
  return json[u'1642590616021670'][u'timeline_stories'][u'nodes']

def get_media_objs(json):
  return json[u'attached_story'][u'attachments'][0][u'media']

def is_relevant_media(json):
  return json[u'owner'][u'id'] == '24983228911'

def video_prop_iter(parsed_json):
  for video in get_video_objs(parsed_json):
    media = get_media_objs(video)
    if not is_relevant_media(media): continue 
    for media_key, unparsed_xml in media.iteritems():
      if not media_key.startswith(u'playlist'): continue
      for xml_elem in xml.etree.ElementTree.fromstring(unparsed_xml).iter(ReprXmlElement):
        if not is_vid(xml_elem): continue
        yield VideoProperties(xml_elem, media_key).index_pair() 

class VideoFinder:
  def __init__(self):
    with open('viewportalresponse2.json','rb') as j:
      parsed_json = json.load(j)
      
    self.video_dct = dict(video_prop_iter(parsed_json))

  def get_video(self, fname):
    return self.video_dct[fname]
