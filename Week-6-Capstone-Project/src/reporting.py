"""Small standard-library DOCX writer with tables and embedded PNG figures."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
for prefix, uri in [("w",W),("r",R),("wp",WP),("a",A),("pic",PIC),("rel",REL),("ct",CT)]: ET.register_namespace(prefix,uri)
q = lambda ns, name: f"{{{ns}}}{name}"


class Report:
    def __init__(self):
        self.root = ET.Element(q(W,"document")); self.body = ET.SubElement(self.root,q(W,"body"))
        self.images=[]; self.counter=0
    def paragraph(self, text, style=None):
        p=ET.SubElement(self.body,q(W,"p"))
        if style:
            ppr=ET.SubElement(p,q(W,"pPr")); ps=ET.SubElement(ppr,q(W,"pStyle")); ps.set(q(W,"val"),style)
        run=ET.SubElement(p,q(W,"r")); t=ET.SubElement(run,q(W,"t")); t.text=str(text)
    def heading(self,text,level=1): self.paragraph(text,f"Heading{level}")
    def table(self,headers,rows):
        tbl=ET.SubElement(self.body,q(W,"tbl")); pr=ET.SubElement(tbl,q(W,"tblPr")); borders=ET.SubElement(pr,q(W,"tblBorders"))
        for edge in ["top","left","bottom","right","insideH","insideV"]:
            ET.SubElement(borders,q(W,edge),{q(W,"val"):"single",q(W,"sz"):"4",q(W,"color"):"9FBAD0"})
        for ri,row in enumerate([headers,*rows]):
            tr=ET.SubElement(tbl,q(W,"tr"))
            for val in row:
                tc=ET.SubElement(tr,q(W,"tc")); p=ET.SubElement(tc,q(W,"p")); run=ET.SubElement(p,q(W,"r"))
                if ri==0: ET.SubElement(run,q(W,"rPr")).append(ET.Element(q(W,"b")))
                ET.SubElement(run,q(W,"t")).text=str(val)
    def image(self,path:Path,caption:str):
        self.counter+=1; rid=f"rIdImage{self.counter}"; name=f"image{self.counter}.png"; data=path.read_bytes(); self.images.append((rid,name,data))
        width=int.from_bytes(data[16:20],"big"); height=int.from_bytes(data[20:24],"big"); cx=5_800_000; cy=int(cx*height/max(width,1))
        p=ET.SubElement(self.body,q(W,"p")); run=ET.SubElement(p,q(W,"r")); drawing=ET.SubElement(run,q(W,"drawing")); inline=ET.SubElement(drawing,q(WP,"inline"))
        ET.SubElement(inline,q(WP,"extent"),{"cx":str(cx),"cy":str(cy)}); ET.SubElement(inline,q(WP,"docPr"),{"id":str(self.counter),"name":name,"descr":caption})
        graphic=ET.SubElement(inline,q(A,"graphic")); gd=ET.SubElement(graphic,q(A,"graphicData"),{"uri":PIC}); pic=ET.SubElement(gd,q(PIC,"pic"))
        nv=ET.SubElement(pic,q(PIC,"nvPicPr")); ET.SubElement(nv,q(PIC,"cNvPr"),{"id":str(self.counter),"name":name}); ET.SubElement(nv,q(PIC,"cNvPicPr"))
        fill=ET.SubElement(pic,q(PIC,"blipFill")); ET.SubElement(fill,q(A,"blip"),{q(R,"embed"):rid}); stretch=ET.SubElement(fill,q(A,"stretch")); ET.SubElement(stretch,q(A,"fillRect"))
        sp=ET.SubElement(pic,q(PIC,"spPr")); xfrm=ET.SubElement(sp,q(A,"xfrm")); ET.SubElement(xfrm,q(A,"off"),{"x":"0","y":"0"}); ET.SubElement(xfrm,q(A,"ext"),{"cx":str(cx),"cy":str(cy)})
        geom=ET.SubElement(sp,q(A,"prstGeom"),{"prst":"rect"}); ET.SubElement(geom,q(A,"avLst")); self.paragraph(caption)
    def save(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        ET.SubElement(self.body,q(W,"sectPr"))
        styles=ET.Element(q(W,"styles")); defaults=ET.SubElement(styles,q(W,"docDefaults")); rpr=ET.SubElement(defaults,q(W,"rPrDefault")); ET.SubElement(rpr,q(W,"rPr"))
        for level in range(1,4):
            style=ET.SubElement(styles,q(W,"style"),{q(W,"type"):"paragraph",q(W,"styleId"):f"Heading{level}"}); ET.SubElement(style,q(W,"name"),{q(W,"val"):f"heading {level}"})
            ppr=ET.SubElement(style,q(W,"pPr")); ET.SubElement(ppr,q(W,"keepNext")); rp=ET.SubElement(style,q(W,"rPr")); ET.SubElement(rp,q(W,"b")); ET.SubElement(rp,q(W,"sz"),{q(W,"val"):str(32-4*level)})
        relroot=ET.Element(q(REL,"Relationships")); ET.SubElement(relroot,q(REL,"Relationship"),{"Id":"rIdStyles","Type":"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles","Target":"styles.xml"})
        docrels=ET.Element(q(REL,"Relationships"))
        for rid,name,_ in self.images: ET.SubElement(docrels,q(REL,"Relationship"),{"Id":rid,"Type":"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image","Target":f"media/{name}"})
        types=ET.Element(q(CT,"Types")); ET.SubElement(types,q(CT,"Default"),{"Extension":"rels","ContentType":"application/vnd.openxmlformats-package.relationships+xml"}); ET.SubElement(types,q(CT,"Default"),{"Extension":"xml","ContentType":"application/xml"}); ET.SubElement(types,q(CT,"Default"),{"Extension":"png","ContentType":"image/png"}); ET.SubElement(types,q(CT,"Override"),{"PartName":"/word/document.xml","ContentType":"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"}); ET.SubElement(types,q(CT,"Override"),{"PartName":"/word/styles.xml","ContentType":"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
        with ZipFile(path,"w",ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",ET.tostring(types,encoding="utf-8",xml_declaration=True)); z.writestr("_rels/.rels",ET.tostring(relroot,encoding="utf-8",xml_declaration=True))
            z.writestr("word/document.xml",ET.tostring(self.root,encoding="utf-8",xml_declaration=True)); z.writestr("word/styles.xml",ET.tostring(styles,encoding="utf-8",xml_declaration=True)); z.writestr("word/_rels/document.xml.rels",ET.tostring(docrels,encoding="utf-8",xml_declaration=True))
            for _,name,data in self.images: z.writestr(f"word/media/{name}",data)
