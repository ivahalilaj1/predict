from wagtail.core import blocks


class BannerSlidesBlock(blocks.StructBlock):
    heading_1 = blocks.CharBlock(classname="full title")
    paragraph_1 = blocks.RichTextBlock()

    heading_2 = blocks.CharBlock(classname="full title")
    paragraph_2 = blocks.RichTextBlock()

    heading_3 = blocks.CharBlock(classname="full title")
    paragraph_3 = blocks.RichTextBlock()

    class Meta:
        template = 'home/blocks/header.html'
        icon = 'cogs'
        label = 'Banner slides'


class FeaturesBlock(blocks.StructBlock):
    title = blocks.CharBlock(classname="full title")

    feature_1 = blocks.CharBlock(classname="full title")
    paragraph_1 = blocks.RichTextBlock()

    feature_2 = blocks.CharBlock(classname="full title")
    paragraph_2 = blocks.RichTextBlock()

    feature_3 = blocks.CharBlock(classname="full title")
    paragraph_3 = blocks.RichTextBlock()

    feature_4 = blocks.CharBlock(classname="full title")
    paragraph_4 = blocks.RichTextBlock()

    class Meta:
        template = 'home/blocks/features.html'
        icon = 'cogs'
        label = 'Features'
