# Generated by Django 2.0.7 on 2018-10-03 20:04

from django.db import migrations
import wagtail.core.blocks
import wagtail.core.fields


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_auto_20180926_1737'),
    ]

    operations = [
        migrations.AddField(
            model_name='homepage',
            name='body',
            field=wagtail.core.fields.StreamField([('bannerslides', wagtail.core.blocks.StructBlock([('heading_1', wagtail.core.blocks.CharBlock(classname='full title')), ('paragraph_1', wagtail.core.blocks.RichTextBlock()), ('heading_2', wagtail.core.blocks.CharBlock(classname='full title')), ('paragraph_2', wagtail.core.blocks.RichTextBlock()), ('heading_3', wagtail.core.blocks.CharBlock(classname='full title')), ('paragraph_3', wagtail.core.blocks.RichTextBlock())])), ('rawhtml', wagtail.core.blocks.RawHTMLBlock())], blank=True, null=True),
        ),
    ]
